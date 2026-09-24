"""Qwen3-VL backend for HERMES streaming KV-cache compression.

The frame-floor strategies live in :mod:`inference.abstract_hermes`.  This
module adapts the Qwen2.5 streaming implementation to Qwen3-VL's interleaved
M-RoPE, Q/K normalization, and DeepStack vision features.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from logzero import logger
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from inference.abstract_hermes import Abstract_Hermes, time_token_tag
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
            text_ids = tokenizer(
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

    def _compute_attention_scores_manually(self, input_ids, past_key_values):
        """Compute pruning attention with Qwen3's normalized Q/K states."""
        device = self.device
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
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attention_weights.append(scores)

        return attention_weights


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
