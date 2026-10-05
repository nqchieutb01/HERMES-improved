"""Qwen3-VL backend for HERMES streaming KV-cache compression.

The frame-floor strategies live in :mod:`inference.abstract_hermes`.  This
module adapts the Qwen2.5 streaming implementation to Qwen3-VL's interleaved
M-RoPE, Q/K normalization, and DeepStack vision features.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from logzero import logger
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from inference.abstract_hermes import (
    Abstract_Hermes,
    is_time_token,
    time_token_group_frame,
    time_token_tag,
)
from inference.qwenvl_hermes import (
    QwenVL_Hermes,
    pad_to_temporal_patch,
    use_linear_patch_embed,
)
from inference.reindex_3d import contiguous_kv
from inference.reindex_qwen3 import (
    apply_rotary_delta_to_keys_only,
    apply_rotary_pos_emb,
    compute_cos_sin_for_positions,
)


def get_qwen3_vl_position_ids(
    video_grid_thw,
    seq_len,
    *,
    offset=0,
    vision_config=None,
    sample_fps=1,
):
    """Build temporal/height/width positions for one streamed video chunk."""
    temporal, height, width = (int(value) for value in video_grid_thw)
    spatial_merge_size = int(
        getattr(vision_config, "spatial_merge_size", 2)
    )
    llm_height = height // spatial_merge_size
    llm_width = width // spatial_merge_size
    expected_tokens = temporal * llm_height * llm_width
    if expected_tokens != int(seq_len):
        raise ValueError(
            "Qwen3-VL video grid does not match encoded token count: "
            f"grid={temporal}x{height}x{width}, "
            f"spatial_merge_size={spatial_merge_size}, "
            f"expected={expected_tokens}, encoded={seq_len}"
        )

    # A temporal grid entry covers ``temporal_patch_size`` source frames.
    temporal_patch_size = float(
        getattr(vision_config, "temporal_patch_size", 2)
    )
    seconds_per_grid = temporal_patch_size / float(sample_fps)
    tokens_per_second = float(
        getattr(vision_config, "tokens_per_second", 1.0)
    )
    temporal_positions = (
        torch.arange(temporal, dtype=torch.float32)
        * seconds_per_grid
        * tokens_per_second
    )
    temporal_positions = temporal_positions.view(-1, 1).expand(
        -1, llm_height * llm_width
    )
    temporal_positions = temporal_positions.flatten() + offset

    height_positions = (
        torch.arange(llm_height, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(temporal, -1, llm_width)
        .flatten()
        + offset
    )
    width_positions = (
        torch.arange(llm_width, dtype=torch.float32)
        .view(1, 1, -1)
        .expand(temporal, llm_height, -1)
        .flatten()
        + offset
    )
    return torch.stack(
        (temporal_positions, height_positions, width_positions)
    )


def get_attention_sequence_ids(position_ids):
    """Return monotonic IDs used only for packed-sequence detection.

    Qwen's temporal M-RoPE coordinate repeats across spatial patches, which
    Transformers would otherwise mistake for several packed documents. The
    actual rotary coordinates are supplied separately through
    ``position_embeddings``.
    """
    batch = position_ids.shape[1]
    seq_len = position_ids.shape[2]
    return torch.arange(
        seq_len,
        device=position_ids.device,
        dtype=position_ids.dtype,
    ).unsqueeze(0).expand(batch, -1)


class Qwen3VL_Hermes(QwenVL_Hermes):
    """Qwen3-VL with HERMES strict shrinking and frame-floor strategies."""

    def get_input_embeddings(self):
        return Qwen3VLForConditionalGeneration.get_input_embeddings(self)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return Qwen3VLForConditionalGeneration.get_video_features(
            self, pixel_values_videos, video_grid_thw
        )

    def _register_forward_hooks(self):
        """Supply layer-specific interleaved M-RoPE after cache shrinking."""

        def make_hook(layer_idx):
            def hook(module, args, kwargs):
                position_ids = self._layer_position_ids.get(layer_idx)
                if position_ids is None:
                    return args, kwargs
                hidden_states = args[0] if args else kwargs["hidden_states"]
                kwargs["position_ids"] = get_attention_sequence_ids(
                    position_ids
                )
                kwargs["position_embeddings"] = self.language_model.rotary_emb(
                    hidden_states, position_ids
                )
                return args, kwargs

            return hook

        for layer_idx, layer in enumerate(self.language_model.layers):
            handle = layer.register_forward_pre_hook(
                make_hook(layer_idx), with_kwargs=True
            )
            self._hook_handles.append(handle)

    def _compute_rotary_cos_sin(self, seq_len, position_ids, dtype, device):
        return compute_cos_sin_for_positions(
            self.language_model, seq_len, position_ids, dtype, device
        )

    def _apply_rotary_to_qk(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    def _apply_rotary_delta_to_keys(self, key_states, cos_delta, sin_delta):
        return apply_rotary_delta_to_keys_only(
            key_states, cos_delta, sin_delta
        )

    def _build_native_video_sequence(
        self, video_features, grid_thw, frame_times, *, first_frame, num_real_frames
    ):
        """Lay out a chunk the way Qwen3-VL was trained to see video.

        Each temporal patch (``temporal_patch_size`` frames) becomes
        ``<t seconds><|vision_start|>[patch tokens]<|vision_end|>``, where ``t``
        is the patch's mean source time. Positions follow Qwen3-VL's
        ``get_rope_index``: text advances all three M-RoPE axes together, and a
        frame's tokens share one temporal index with row/column offsets.
        Returns embeddings, positions relative to the cache end (3 x L),
        the visual-token mask and each token's source frame id
        (a time_token_tag of the group's first frame for the timestamp/marker text).
        """
        temporal, height, width = (int(v) for v in grid_thw)
        vision_config = self.config.vision_config
        merge = int(getattr(vision_config, "spatial_merge_size", 2))
        patch = int(getattr(vision_config, "temporal_patch_size", 2))
        llm_h, llm_w = height // merge, width // merge
        frame_len = llm_h * llm_w
        if temporal * frame_len != video_features.shape[0]:
            raise ValueError(
                f"Qwen3-VL grid {grid_thw} does not match "
                f"{video_features.shape[0]} encoded video tokens"
            )
        times = list(frame_times) + [frame_times[-1]] * (
            temporal * patch - len(frame_times)
        )
        tokenizer = self.processor.tokenizer
        start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        embed = self.get_input_embeddings()
        device = video_features.device
        rows = torch.arange(llm_h, device=device).repeat_interleave(llm_w)
        cols = torch.arange(llm_w, device=device).repeat(llm_h)

        pieces, positions, masks, frame_ids = [], [], [], []
        cursor = 0

        def add_text(ids, group_first_frame):
            nonlocal cursor
            ids = torch.tensor(ids, device=device)
            pieces.append(embed(ids).to(video_features.dtype))
            positions.append(
                torch.arange(cursor, cursor + len(ids), device=device)
                .float()
                .expand(3, -1)
            )
            masks.append(torch.zeros(len(ids), dtype=torch.bool, device=device))
            # Tagged as text of this group so keep_time_tokens can find them.
            frame_ids.extend([time_token_tag(group_first_frame)] * len(ids))
            cursor += len(ids)

        for group in range(temporal):
            real = [
                first_frame + min(group * patch + k, num_real_frames - 1)
                for k in range(patch)
            ]
            stamp = (times[group * patch] + times[group * patch + patch - 1]) / 2
            # Diagnostics: shift every timestamp by time_offset seconds, or drop the text.
            stamp += float(getattr(self, "time_offset", 0.0))
            text_ids = [] if getattr(self, "drop_timestamps", False) else tokenizer(
                f"<{stamp:.1f} seconds>", add_special_tokens=False
            ).input_ids
            add_text(text_ids + [start_id], real[0])

            pieces.append(video_features[group * frame_len:(group + 1) * frame_len])
            positions.append(
                torch.stack(
                    (
                        torch.full_like(rows, cursor),
                        rows + cursor,
                        cols + cursor,
                    )
                ).float()
            )
            masks.append(torch.ones(frame_len, dtype=torch.bool, device=device))
            frame_ids.extend(real[i * patch // frame_len] for i in range(frame_len))
            cursor += max(llm_h, llm_w)

            add_text([end_id], real[0])

        return (
            torch.cat(pieces, dim=0).unsqueeze(0),
            torch.cat(positions, dim=1),
            torch.cat(masks).unsqueeze(0),
            torch.tensor(frame_ids, dtype=torch.long),
        )

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk):
        if video_chunk is None or (
            hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0
        ):
            return

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        num_frames = video_chunk.shape[0]
        frame_times = self._consume_frame_times(num_frames)
        temporal_patch_size = int(
            getattr(self.config.vision_config, "temporal_patch_size", 2)
        )
        processor_chunk = pad_to_temporal_patch(video_chunk, temporal_patch_size)
        # HERMES already sampled frames at sample_fps; stop the Qwen3 processor
        # from resampling them again (it assumes 24 fps and keeps ~1 in 4).
        video_input = self.processor(
            text=[""],
            videos=processor_chunk,
            do_sample_frames=False,
            return_tensors="pt",
        ).to(self.device, self.dtype)
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        if self.total_processed_frames == 0:
            t, h, w = video_grid_thw[0].tolist()
            logger.info(f"video grid {t}x{h}x{w}: {h * w // 4} visual tokens per temporal group")
        video_feature_groups, deepstack_features = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_features = torch.cat(video_feature_groups, dim=0)
        deepstack_features = [
            feature.to(device=self.device, dtype=video_features.dtype)
            for feature in deepstack_features
        ]
        inputs_embeds, rel_pos_ids, visual_pos_masks, frame_ids = (
            self._build_native_video_sequence(
                video_features,
                video_grid_thw[0].tolist(),
                frame_times,
                first_frame=self.total_processed_frames,
                num_real_frames=num_frames,
            )
        )

        self._ensure_dynamic_cache()
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        batch = inputs_embeds.shape[0]
        base_offset = global_offset_per_layer[0]
        grid_pos_ids = rel_pos_ids + base_offset

        self._layer_position_ids.clear()
        for layer_idx, layer_offset in enumerate(global_offset_per_layer):
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos += layer_offset - base_offset
            self._layer_position_ids[layer_idx] = (
                self._build_position_ids_3d_for_vision(
                    current_layer_pos, batch
                )
            )

        default_position_ids = self._build_position_ids_3d_for_vision(
            grid_pos_ids, batch
        )
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=default_position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_features,
        )
        self.kv_cache = output.past_key_values
        contiguous_kv(self.kv_cache)

        for layer_idx, layer_offset in enumerate(global_offset_per_layer):
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos += layer_offset - base_offset
            self._append_position_ids_layer_explicit(
                layer_idx, current_layer_pos
            )

        if self.token_provenance_enabled:
            self._append_token_frame_ids(frame_ids)

        self.last_encoded_frames = num_frames
        self.total_processed_frames += num_frames
        self._layer_position_ids.clear()
        torch.cuda.empty_cache()

    # ----- Memory snapshots (for decoding against a counterfactual memory) -----
    _STATE = ("kv_cache", "_position_ids_cache", "_token_frame_ids_per_layer", "_token_frame_summary_scores_per_layer",
              "_frame_summary_specs_per_layer", "_encoded_tokens_per_frame", "visual_start_idx", "conv_history")

    def memory_snapshot(self):
        state = {k: getattr(self, k, None) for k in self._STATE}
        state["_position_ids_cache"] = list(self._position_ids_cache)
        state["conv_history"] = list(self.conv_history or [])
        return state

    def memory_restore(self, state):
        for k, v in state.items():
            setattr(self, k, list(v) if k in ("_position_ids_cache", "conv_history") and v is not None else v)

    def _forward_text(self, ids):
        """Feed text token ids into the current memory (prefill or one decoding step); returns last-token logits."""
        offsets = self._get_next_global_offset_per_layer()
        q_len = ids.shape[1]
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(offsets[layer_idx], q_len, 1)
        out = self.language_model(inputs_embeds=self.get_input_embeddings()(ids), use_cache=True,
                                  past_key_values=self.kv_cache,
                                  position_ids=self._build_position_ids_3d_for_text(offsets[0], q_len, 1))
        self.kv_cache = out.past_key_values
        for layer_idx in range(self.num_layers):
            o = offsets[layer_idx]
            self._append_position_ids_layer(layer_idx, [o, o, o], q_len)
        self._layer_position_ids.clear()
        return self.lm_head(out.last_hidden_state)[0, -1].float()

    @staticmethod
    def _in_time_slot(text, scope):
        """Whether the next token writes a time value (the slot the temporal prior leaks into).

        "time": inside a `Seen: <t>` moment or an `Event: <start> - <end>` line before its description, or on the
        `Time: [...]` line.
        "time+answer": also anywhere on the `Answer:` line (durations and counts are stated there).
        """
        line = text.rsplit("\n", 1)[-1].lstrip()
        if line.startswith(("Seen:", "Event:")):
            return "," not in line and "second" not in line
        if line.startswith("Time:"):
            return "]" not in line
        return scope == "time+answer" and line.startswith("Answer:")

    @torch.inference_mode()
    def contrastive_answering(self, prompt, negative_state, alpha=1.0, beta=0.1, max_new_tokens=384,
                              repetition_penalty=1.1, scope="all", adaptive=False, rule="pmi", trace=None):
        """Greedy decoding contrasted against a counterfactual memory (VCD-style, Leng et al. CVPR 2024).

        Both memories receive the same prompt and the same generated tokens. At each step the next token maximises
        (1 + alpha) * log p(. | real memory) - alpha * log p(. | counterfactual memory), restricted to tokens whose
        probability under the real memory is at least beta times the most likely one (adaptive plausibility).
        Tokens the model would produce regardless of the real memory are thereby discounted.

        ``rule="against"`` keeps only the penalty half: log p(. | M) - alpha * max(0, log p(. | C) - log p(. | M)).
        log p(y | M) - log p(y | C) is the log-likelihood ratio of the real memory for token y; the rule discounts
        tokens the memory provides evidence against (ratio < 1) and leaves tokens it supports ranked by p(. | M),
        without rewarding tokens merely for being improbable under the counterfactual. ``trace`` (a list) receives,
        at every contrasted step, the top tokens under the real memory with both log-probabilities.
        """
        tok = self.processor.tokenizer
        ids = torch.as_tensor([tok(prompt).input_ids], device=self.device)
        self._ensure_dynamic_cache()
        before = self._get_cache_seq_len_per_layer()
        logit_pos = self._forward_text(ids)
        positive = self.memory_snapshot()
        self.memory_restore(negative_state)
        self._ensure_dynamic_cache()
        logit_neg = self._forward_text(ids)
        negative = self.memory_snapshot()
        out_ids = []
        for _ in range(max_new_tokens):
            lp = torch.log_softmax(logit_pos, -1)
            ln = torch.log_softmax(logit_neg, -1)
            if scope == "all" or self._in_time_slot(tok.decode(out_ids), scope):
                # Confidence-adaptive strength: contrast fades where the real memory is already confident.
                a = alpha * (1.0 - float(lp.max().exp())) if adaptive else alpha
                if rule == "against":
                    score = lp - a * torch.clamp(ln - lp, min=0.0)
                else:
                    score = (1.0 + a) * lp - a * ln
                score = score.masked_fill(lp < lp.max() + math.log(beta), float("-inf"))
            else:
                score = lp.clone()  # outside the time slots: plain greedy decoding on the real memory
            for t in set(out_ids):  # repetition penalty on the contrasted score (as on logits in greedy decoding)
                score[t] = score[t] - math.log(repetition_penalty)
            token = int(torch.argmax(score))
            if trace is not None and self._in_time_slot(tok.decode(out_ids), "time"):
                top = torch.topk(lp, 20).indices.tolist()
                trace.append({"step": len(out_ids), "prefix": tok.decode(out_ids)[-40:], "chosen": tok.decode([token]),
                              "top": [[tok.decode([t]), round(float(lp[t]), 3), round(float(ln[t]), 3)] for t in top]})
            out_ids.append(token)
            if token == tok.eos_token_id:
                break
            step = torch.as_tensor([[token]], device=self.device)
            self.memory_restore(positive)
            logit_pos = self._forward_text(step)
            positive = self.memory_snapshot()
            self.memory_restore(negative)
            logit_neg = self._forward_text(step)
            negative = self.memory_snapshot()
        self.memory_restore(positive)
        # As in question_answering: drop the prompt and answer tokens so the memory is unchanged for later questions.
        self._truncate_kv_cache(before)
        for layer_idx in range(self.num_layers):
            pos = self._position_ids_cache[layer_idx]
            if pos is not None and pos.shape[1] > before[layer_idx]:
                self._position_ids_cache[layer_idx] = pos[:, :before[layer_idx]].contiguous()
        return tok.decode(out_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                          clean_up_tokenization_spaces=True)

    def question_answering(self, input_text, *args, **kwargs):
        """Answer generation; optionally with visual-attention rebalancing.

        With ``visual_attention_gain`` = gamma != 1, every attention call during the prompt prefill and decoding
        adds log(gamma) to the logits of visual-memory tokens in ``visual_attention_layers`` (all layers if None),
        so their attention mass is multiplied by gamma before renormalisation (PAI-style, Liu et al. ECCV 2024).
        """
        gain = float(getattr(self, "visual_attention_gain", 1.0))
        if gain == 1.0 or self._token_frame_ids_per_layer is None:
            return super().question_answering(input_text, *args, **kwargs)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        layers = getattr(self, "visual_attention_layers", None)
        layers = set(range(self.num_layers)) if layers is None else set(int(l) for l in layers)
        log_gain = math.log(gain)
        visual = {l: (self._token_frame_ids_per_layer[l] >= 0).to(self.device) for l in layers}

        def biased_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **_):
            k = repeat_kv(key, module.num_key_value_groups)
            v = repeat_kv(value, module.num_key_value_groups)
            scores = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
            q_len, kv_len = scores.shape[-2], scores.shape[-1]
            if attention_mask is not None and attention_mask.dim() == 4:
                scores = scores + attention_mask[:, :, :, :kv_len].float()
            elif q_len > 1:
                causal = torch.ones(q_len, q_len, dtype=torch.bool, device=scores.device).triu(1)
                scores[..., kv_len - q_len:] = scores[..., kv_len - q_len:].masked_fill(causal, float("-inf"))
            mask = visual.get(module.layer_idx)
            if mask is not None:
                n = min(mask.numel(), kv_len)
                scores[..., :n] = scores[..., :n] + log_gain * mask[:n].to(scores.dtype)
            weights = torch.softmax(scores, dim=-1).to(v.dtype)
            return torch.matmul(weights, v).transpose(1, 2).contiguous(), None

        impl = self.language_model.config._attn_implementation
        ALL_ATTENTION_FUNCTIONS[impl] = biased_attention
        try:
            return super().question_answering(input_text, *args, **kwargs)
        finally:
            del ALL_ATTENTION_FUNCTIONS[impl]

    @torch.no_grad()
    def encode_timestamp_text(self, times):
        """Diagnostics: append timestamp text only ("<t seconds>" per time, no frame) to the cache.

        Used by the evidence-window oracle to mark video time outside the shown segment. Text advances all
        three M-RoPE axes together, as in _build_native_video_sequence; tokens are tagged as text (-1).
        """
        if not times:
            return
        tokenizer = self.processor.tokenizer
        ids = []
        for t in times:
            ids += tokenizer(f"<{t:.1f} seconds>", add_special_tokens=False).input_ids
        ids = torch.tensor(ids, device=self.device)
        inputs_embeds = self.get_input_embeddings()(ids).to(self.dtype).unsqueeze(0)
        rel_pos_ids = torch.arange(len(ids), device=self.device).float().expand(3, -1)

        self._ensure_dynamic_cache()
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        base_offset = global_offset_per_layer[0]
        grid_pos_ids = rel_pos_ids + base_offset
        self._layer_position_ids.clear()
        for layer_idx, layer_offset in enumerate(global_offset_per_layer):
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos += layer_offset - base_offset
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_vision(current_layer_pos, 1)
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=self._build_position_ids_3d_for_vision(grid_pos_ids, 1),
        )
        self.kv_cache = output.past_key_values
        contiguous_kv(self.kv_cache)
        for layer_idx, layer_offset in enumerate(global_offset_per_layer):
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos += layer_offset - base_offset
            self._append_position_ids_layer_explicit(layer_idx, current_layer_pos)
        if self.token_provenance_enabled:
            self._append_token_frame_ids([-1] * len(ids))
        self._layer_position_ids.clear()

    def _compute_attention_scores_manually(self, input_ids, past_key_values, offsets=None, reduce=None):
        """Compute pruning attention with Qwen3's normalized Q/K states.

        Diagnostics: ``offsets`` overrides the text positions; ``reduce(layer_idx, scores)`` replaces each
        layer's full attention map by its return value (keeps memory flat for long query blocks).
        """
        device = self.device
        if offsets is None:
            offsets = self._get_next_global_offset_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]
        hidden_states = self.get_input_embeddings()(input_ids)
        config = self.language_model.config
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        attention_weights = []
        exact = getattr(self, "exact_attention", False)

        for layer_idx, layer in enumerate(self.language_model.layers):
            past_key, past_value = past_key_values[layer_idx]
            position_ids = self._build_position_ids_3d_for_text(
                offsets[layer_idx], q_len, batch
            )
            normalized = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            projected_shape = (batch, q_len, -1, head_dim)
            query_states = attention.q_proj(normalized).view(
                projected_shape
            )
            key_states = attention.k_proj(normalized).view(projected_shape)
            value_states = attention.v_proj(normalized).view(
                projected_shape
            )
            query_states = attention.q_norm(query_states).transpose(1, 2)
            key_states = attention.k_norm(key_states).transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = self._compute_rotary_cos_sin(
                q_len, position_ids, hidden_states.dtype, device
            )
            query_states, key_states = self._apply_rotary_to_qk(
                query_states, key_states, cos, sin
            )
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)

            if config.num_key_value_heads != config.num_attention_heads:
                repeats = (
                    config.num_attention_heads
                    // config.num_key_value_heads
                )
                key_states = torch.repeat_interleave(
                    key_states, repeats, dim=1
                )
                value_states = torch.repeat_interleave(
                    value_states, repeats, dim=1
                )

            scores = torch.matmul(
                query_states.float(), key_states.float().transpose(-2, -1)
            ) * attention.scaling
            if exact:
                # Causal mask within the query block (the cached prefix is fully visible).
                past = past_key.shape[2]
                block = torch.ones(q_len, q_len, dtype=torch.bool, device=device).triu(1)
                scores[..., past:] = scores[..., past:].masked_fill(block, float("-inf"))
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attention_weights.append(scores if reduce is None else reduce(layer_idx, scores))
            if exact:
                # Propagate the query tokens through the full layer (attention output, residual,
                # MLP), so the next layer's queries come from real hidden states. The default path
                # reuses the input embeddings at every layer (HERMES's original approximation).
                attended = torch.matmul(scores, value_states).transpose(1, 2).reshape(batch, q_len, -1)
                hidden_states = hidden_states + attention.o_proj(attended)
                hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

        return attention_weights

    def head_relevance(self, text, heads, first_frame=0):
        """Per-frame attention mass from selected (layer, head) pairs while reading ``text`` over the current cache.

        Exact propagation through all layers; for each listed head, attention is averaged over the text tokens
        and summed over each frame's (kept) visual tokens, then averaged over the heads. Returns
        {"mass": {frame: m}, "tokens": {frame: kept visual tokens}} with frame ids relative to ``first_frame``.
        """
        ids = torch.as_tensor([self.processor.tokenizer(text).input_ids], device=self.device)
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(int(l), []).append(int(h))
        kv_len = self._get_cache_seq_len_per_layer()[0]

        def reduce(layer_idx, scores):
            if layer_idx not in by_layer:
                return None
            m = scores[0, by_layer[layer_idx], :, :kv_len].float().mean(dim=1)  # heads x kv
            return m.sum(dim=0)  # summed over this layer's selected heads

        exact_before = getattr(self, "exact_attention", False)
        self.exact_attention = True
        try:
            with torch.no_grad():
                per_layer = self._compute_attention_scores_manually(ids, self.kv_cache, reduce=reduce)
        finally:
            self.exact_attention = exact_before
        mass, tokens = {}, {}
        for layer_idx, m in enumerate(per_layer):
            if m is None:
                continue
            frame_ids = self._token_frame_ids_per_layer[layer_idx][:kv_len].to(m.device)
            for frame in torch.unique(frame_ids[frame_ids >= 0]).tolist():
                sel = frame_ids == frame
                mass[frame - first_frame] = mass.get(frame - first_frame, 0.0) + float(m[sel].sum()) / len(heads)
                tokens[frame - first_frame] = int(sel.sum())
        return {"mass": {k: round(v, 6) for k, v in sorted(mass.items())}, "tokens": dict(sorted(tokens.items()))}

    def question_attention_profile(self, question, first_frame=0):
        """Diagnostics: attention from the real question to every frame of the current cache.

        Runs the question text through all layers with exact propagation and returns, per layer
        band (early / mid / late thirds), the attention mass on each frame's visual tokens and on
        each temporal group's timestamp text, averaged over heads and question tokens. Frame ids are
        relative to ``first_frame``.
        """
        ids = torch.as_tensor([self.processor.tokenizer(question).input_ids], device=self.device)
        exact_before = getattr(self, "exact_attention", False)
        self.exact_attention = True
        try:
            with torch.no_grad():
                weights = self._compute_attention_scores_manually(ids, self.kv_cache)
        finally:
            self.exact_attention = exact_before
        num_layers = len(weights)
        bands = [range(0, num_layers // 3), range(num_layers // 3, 2 * num_layers // 3),
                 range(2 * num_layers // 3, num_layers)]
        visual, stamps = {}, {}
        for band_idx, band in enumerate(bands):
            for layer_idx in band:
                mass = weights[layer_idx][0].float().mean(dim=(0, 1))  # over heads and query tokens
                ids_layer = self._token_frame_ids_per_layer[layer_idx].to(mass.device)
                mass = mass[: ids_layer.numel()]
                for frame in torch.unique(ids_layer[ids_layer >= 0]).tolist():
                    v = visual.setdefault(frame - first_frame, [0.0, 0.0, 0.0])
                    v[band_idx] += float(mass[ids_layer == frame].sum()) / len(band)
                time_ids = is_time_token(ids_layer)
                if time_ids.any():
                    groups = time_token_group_frame(ids_layer[time_ids])
                    group_mass = mass[time_ids]
                    for frame in torch.unique(groups).tolist():
                        s = stamps.setdefault(int(frame) - first_frame, [0.0, 0.0, 0.0])
                        s[band_idx] += float(group_mass[groups == frame].sum()) / len(band)
        round3 = lambda d: {int(k): [round(x, 6) for x in v] for k, v in sorted(d.items())}
        return {"visual": round3(visual), "timestamps": round3(stamps)}

    def _gold_masks(self, layer_idx, kv_len, first_frame, gold_frames):
        ids = self._token_frame_ids_per_layer[layer_idx][:kv_len].to(self.device)
        gold = torch.as_tensor(sorted(first_frame + g for g in gold_frames), dtype=ids.dtype, device=self.device)
        return ids >= 0, torch.isin(ids, gold)

    def answer_attention_profile(self, prompt_text, answer_text, cache_len, offsets, first_frame, gold_frames):
        """Diagnostics: attention to the gold-interval frames while reading the prompt and while writing
        the answer and its timestamps (teacher-forced replay of the model's own output).

        The cache is cropped back to ``cache_len`` (its length before answering) and the prompt plus the
        generated answer are replayed with exact propagation from the pre-answer ``offsets``. For three row
        groups (prompt, answer text, text from "Time:" on) returns, per layer and head, the share of
        visual attention on gold frames, and per layer the share of all attention that goes to video.
        """
        tok = self.processor.tokenizer
        before_time = answer_text.split("Time:")[0]
        n_prompt = len(tok(prompt_text).input_ids)
        n_answer = len(tok(prompt_text + before_time).input_ids)
        ids = tok(prompt_text + answer_text).input_ids
        groups = {"prompt": (0, n_prompt), "answer": (n_prompt, n_answer), "time": (n_answer, len(ids))}
        groups = {k: v for k, v in groups.items() if v[1] > v[0]}
        self.kv_cache.crop(int(cache_len))
        kv_len = int(cache_len)

        def reduce(layer_idx, scores):
            vis, gold = self._gold_masks(layer_idx, kv_len, first_frame, gold_frames)
            out = {}
            for name, (a, b) in groups.items():
                m = scores[0, :, a:b, :kv_len].float().mean(dim=1)  # heads x kv
                visual_mass = m[:, vis].sum(-1)
                out[name] = {"gold_share": (m[:, gold].sum(-1) / visual_mass.clamp_min(1e-9)).tolist(),
                             "video_share": float(visual_mass.mean() / scores[0, :, a:b].float().sum(-1).mean())}
            return out

        exact_before = getattr(self, "exact_attention", False)
        self.exact_attention = True
        try:
            with torch.no_grad():
                per_layer = self._compute_attention_scores_manually(
                    torch.as_tensor([ids], device=self.device), self.kv_cache, offsets=offsets, reduce=reduce)
        finally:
            self.exact_attention = exact_before
        r3 = lambda xs: [round(x, 4) for x in xs]
        return {name: {"gold_share": [r3(layer[name]["gold_share"]) for layer in per_layer],
                       "video_share": [round(layer[name]["video_share"], 4) for layer in per_layer]}
                for name in groups}


def load_model(
    model_path="Qwen/Qwen3-VL-8B-Instruct",
    n_init=None,
    kv_size=None,
    streaming=True,
    device="cuda",
    sample_fps=1,
):
    """Load Qwen3-VL in the dedicated Qwen environment."""
    attention_backend = os.environ.get(
        "HERMES_QWEN3_ATTENTION_BACKEND", "sdpa"
    )
    if attention_backend not in {"sdpa", "flash_attention_2"}:
        raise ValueError(
            "HERMES_QWEN3_ATTENTION_BACKEND must be 'sdpa' or "
            f"'flash_attention_2', got {attention_backend!r}"
        )
    processor = AutoProcessor.from_pretrained(model_path)
    system_prompt = (
        "<|im_start|>system\nYou are a helpful assistant."
        "<|im_end|>\n<|im_start|>user\n"
    )
    init_prompt_ids = processor.tokenizer(
        system_prompt, return_tensors="pt"
    ).input_ids.to(device)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    )
    use_linear_patch_embed(base_model.visual)

    model = Qwen3VL_Hermes.__new__(Qwen3VL_Hermes)
    model.__dict__ = base_model.__dict__.copy()
    Abstract_Hermes.__init__(
        model,
        processor,
        init_prompt_ids.tolist(),
        kv_size,
    )
    model.streaming = streaming
    model.sample_fps = sample_fps
    model.num_layers = base_model.language_model.config.num_hidden_layers
    model._position_ids_cache = [None for _ in range(model.num_layers)]
    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(
        model.num_layers * model.short_term_ratio
    )
    model.long_term_threshold = int(
        model.num_layers * (1 - model.long_term_ratio)
    )
    model.total_processed_frames = 0
    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()

    logger.info(
        "n_init: %s",
        init_prompt_ids.shape[1] if n_init is None else n_init,
    )
    logger.info("kv_size: %s", kv_size)
    logger.info("attention backend: %s", attention_backend)
    model.eval()
    return model, processor


__all__ = [
    "Qwen3VL_Hermes",
    "get_attention_sequence_ids",
    "get_qwen3_vl_position_ids",
    "load_model",
]
