import csv
from collections import Counter
from pathlib import Path

import torch

class Abstract_Hermes:
    kv_cache = None
    FRAME_SUMMARY_STRATEGIES = (
        "mean",
        "top_patch",
        "top_attention_patch",
        "attention_weighted",
        "softmax_score_weighted",
    )

    def __init__(self, processor, init_prompt_ids, kv_size):
        self.processor = processor
        self.init_prompt_ids = init_prompt_ids
        self.kv_size = kv_size
        self.last_encoded_frames = 0
        self.visual_start_idx = 14
        self.conv_history = []

        # Optional provenance tracking for inspecting how visual tokens survive
        # HERMES compression. Frame ids live on CPU to avoid adding GPU memory
        # pressure during inference.
        self.token_trace_path = None
        self.token_trace_rows = []
        self._token_trace_video_id = "unknown"
        self._token_trace_event = 0
        self._token_frame_ids_per_layer = None
        self._token_frame_summary_scores_per_layer = None
        self.min_tokens_per_frame = 0
        self.frame_summary_strategy = "mean"
        self.frame_summary_temperature = 0.1
        self.frame_summary_mode = False
        self._frame_summary_specs_per_layer = None
        self.token_trace_verbose = False

    def enable_token_trace(self, path):
        """Enable CSV tracing of retained visual tokens by frame and layer."""
        self.token_trace_path = str(path)

    def set_token_trace_verbose(self, value):
        """Enable detailed per-event token-retention messages."""
        self.token_trace_verbose = bool(value)

    def set_token_trace_video(self, video_id):
        """Set the video id used for subsequent token-retention rows."""
        self._token_trace_video_id = str(video_id)
        self._token_trace_event = 0

    @property
    def token_trace_enabled(self):
        return self.token_trace_path is not None

    @property
    def token_provenance_enabled(self):
        return self.token_trace_enabled or self.min_tokens_per_frame > 0

    def set_min_tokens_per_frame(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("min_tokens_per_frame must be nonnegative")
        self.min_tokens_per_frame = value
        self._refresh_frame_summary_mode()

    def set_frame_summary_strategy(self, strategy, temperature=0.1):
        """Choose how a frame with no surviving patch is represented."""
        strategy = str(strategy)
        if strategy not in self.FRAME_SUMMARY_STRATEGIES:
            choices = ", ".join(self.FRAME_SUMMARY_STRATEGIES)
            raise ValueError(
                f"Unknown frame summary strategy {strategy!r}; choose from {choices}"
            )
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("frame_summary_temperature must be positive")
        self.frame_summary_strategy = strategy
        self.frame_summary_temperature = temperature
        self._refresh_frame_summary_mode()

    def _refresh_frame_summary_mode(self):
        # top_patch satisfies k=1 by retaining a real source token. The other
        # strategies synthesize one pooled token for an otherwise missing frame.
        self.frame_summary_mode = (
            self.min_tokens_per_frame == 1
            and self.frame_summary_strategy
            not in {"top_patch", "top_attention_patch"}
        )

    def _reset_token_frame_ids(self):
        """Forget frame provenance when starting a new video."""
        self._token_frame_ids_per_layer = None
        self._token_frame_summary_scores_per_layer = None
        self._frame_summary_specs_per_layer = None

    def _initialize_token_frame_ids(self, cache_lengths):
        """Initialize provenance for the text-only cache (-1 means text)."""
        if not self.token_provenance_enabled:
            return
        self._token_frame_ids_per_layer = [
            torch.full((length,), -1, dtype=torch.long)
            for length in cache_lengths
        ]
        self._token_frame_summary_scores_per_layer = [
            torch.full((length,), float("nan"), dtype=torch.float32)
            for length in cache_lengths
        ]

    def _append_token_frame_ids(self, frame_ids):
        """Append source frame ids for the visual tokens just encoded."""
        if not self.token_provenance_enabled:
            return
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device="cpu")
        if self._token_frame_ids_per_layer is None:
            raise RuntimeError("Token-frame provenance was not initialized")
        self._token_frame_ids_per_layer = [
            torch.cat((ids, frame_ids), dim=0)
            for ids in self._token_frame_ids_per_layer
        ]
        self._token_frame_summary_scores_per_layer = [
            torch.cat(
                (
                    scores,
                    torch.full(
                        (frame_ids.numel(),),
                        float("nan"),
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
            for scores in self._token_frame_summary_scores_per_layer
        ]

    def _frame_ids_for_video_tokens(self, num_frames, num_tokens, first_frame):
        """Map visual tokens to input frame ids.

        Most processors produce a fixed number of tokens per frame. The
        proportional fallback also handles processors that merge frames, in
        which case a token is attributed to the first frame in its group.
        """
        if num_frames <= 0 or num_tokens <= 0:
            return torch.empty(0, dtype=torch.long)
        token_indices = torch.arange(num_tokens, dtype=torch.long)
        frame_indices = (token_indices * num_frames // num_tokens).clamp(
            max=num_frames - 1
        )
        return frame_indices + int(first_frame)

    def _select_indices_with_frame_minimum(
        self, scores, frame_ids, budget, frame_floor_scores=None
    ):
        """Select the normal budget, then top up frames below the floor.

        ``budget`` is the normal visual-token budget for one layer. The
        initial selection is exactly the normal top-k selection. If a frame
        has fewer than ``min_tokens_per_frame`` selected tokens, the highest-
        scoring unselected tokens from that frame are appended. Therefore the
        returned effective budget can be larger than the configured budget.
        """
        budget = min(max(int(budget), 0), scores.numel())
        min_tokens = int(self.min_tokens_per_frame)
        if min_tokens <= 0:
            return torch.topk(scores, budget, sorted=False).indices, budget
        if frame_ids is None or frame_ids.numel() != scores.numel():
            raise RuntimeError(
                "Frame provenance is unavailable or misaligned; cannot enforce "
                "min_tokens_per_frame"
            )
        if frame_floor_scores is None:
            frame_floor_scores = scores
        elif frame_floor_scores.numel() != scores.numel():
            raise RuntimeError(
                "Frame-floor scores do not match selection scores: "
                f"{frame_floor_scores.numel()} != {scores.numel()}"
            )

        if self.frame_summary_mode:
            # For k=1, a missing frame is represented by a synthetic mean
            # token during cache shrinking. Do not spend the normal budget on
            # arbitrary per-frame patch top-ups here.
            return torch.topk(scores, budget, sorted=False).indices, budget

        # First select the normal kv_size tokens. Summary tokens use -2 and
        # text tokens are outside this visual slice; only nonnegative ids
        # participate in the per-frame top-up.
        selected_indices = torch.topk(scores, budget, sorted=False).indices
        frame_ids = frame_ids.to(device=scores.device, dtype=torch.long)
        frame_values = torch.unique(frame_ids[frame_ids >= 0], sorted=True)
        selected_mask = torch.zeros(
            scores.numel(), device=scores.device, dtype=torch.bool
        )
        selected_mask[selected_indices] = True
        additions = []
        for frame_id in frame_values.tolist():
            frame_indices = torch.nonzero(frame_ids == frame_id, as_tuple=False).flatten()
            selected_count = int(selected_mask[frame_indices].sum().item())
            missing_count = min_tokens - selected_count
            if missing_count <= 0:
                continue

            unselected = frame_indices[~selected_mask[frame_indices]]
            missing_count = min(missing_count, unselected.numel())
            if missing_count > 0:
                unselected_scores = frame_floor_scores.index_select(0, unselected)
                top_up = torch.topk(
                    unselected_scores, missing_count, sorted=False
                ).indices
                additions.append(unselected.index_select(0, top_up))
                selected_mask[unselected.index_select(0, top_up)] = True

        if additions:
            selected_indices = torch.cat((selected_indices, *additions))

        selected_indices = torch.unique(selected_indices, sorted=True)
        return selected_indices, selected_indices.numel()

    def _equalize_keep_indices_by_layer(
        self,
        keep_indices_all_layers,
        layer_scores,
        layer_configs,
        layer_frame_ids=None,
    ):
        """Make every layer's post-pruning physical cache length identical.

        The decoder expects one causal-mask width for all layers. Per-layer
        frame top-ups and long-term summary tokens can otherwise produce
        different physical lengths. Add the highest-scoring remaining visual
        tokens to shorter layers; this only increases retention and preserves
        the configured per-frame minimum.
        """
        current_lengths = self._get_cache_seq_len_per_layer()

        def missing_summary_count(layer_idx, keep):
            if not self.frame_summary_mode or layer_frame_ids is None:
                return 0
            score = layer_scores[layer_idx]
            start_idx = layer_configs[layer_idx]["visual_start_idx"]
            frame_ids = layer_frame_ids[layer_idx][
                start_idx:start_idx + score.numel()
            ].to(device=score.device, dtype=torch.long)
            all_frames = torch.unique(frame_ids[frame_ids >= 0], sorted=True)
            if all_frames.numel() == 0:
                return 0
            selected_visual = keep[
                (keep >= start_idx) & (keep < start_idx + score.numel())
            ] - start_idx
            if selected_visual.numel() == 0:
                return int(all_frames.numel())
            selected_frames = torch.unique(
                frame_ids.index_select(0, selected_visual), sorted=True
            )
            selected_frames = selected_frames[selected_frames >= 0]
            return int((~torch.isin(all_frames, selected_frames)).sum().item())

        physical_lengths = []
        for layer_idx, keep_indices in enumerate(keep_indices_all_layers):
            keep = torch.as_tensor(
                keep_indices,
                device=layer_scores[layer_idx].device,
                dtype=torch.long,
            )
            keep_count = len(keep_indices)
            has_summary = (
                layer_idx >= self.long_term_threshold
                and keep_count < current_lengths[layer_idx]
            )
            physical_lengths.append(
                keep_count
                + missing_summary_count(layer_idx, keep)
                + int(has_summary)
            )

        target_length = max(physical_lengths)
        if len(set(physical_lengths)) == 1:
            return keep_indices_all_layers

        equalized = []
        for layer_idx, keep_indices in enumerate(keep_indices_all_layers):
            score = layer_scores[layer_idx]
            start_idx = layer_configs[layer_idx]["visual_start_idx"]
            keep = torch.as_tensor(
                keep_indices, device=score.device, dtype=torch.long
            )
            selected_visual = keep[
                (keep >= start_idx) & (keep < start_idx + score.numel())
            ]
            selected_mask = torch.zeros(
                score.numel(), device=score.device, dtype=torch.bool
            )
            selected_mask[selected_visual - start_idx] = True

            while True:
                summary_count = missing_summary_count(layer_idx, keep)
                has_summary = (
                    layer_idx >= self.long_term_threshold
                    and keep.numel() < current_lengths[layer_idx]
                )
                physical_length = (
                    int(keep.numel()) + summary_count + int(has_summary)
                )
                needed = target_length - physical_length
                if needed <= 0:
                    break

                candidates = torch.nonzero(~selected_mask, as_tuple=False).flatten()
                if candidates.numel() == 0:
                    raise RuntimeError(
                        "Cannot equalize KV-cache lengths: layer "
                        f"{layer_idx} needs {needed} more visual token(s), "
                        f"but only {candidates.numel()} remain"
                    )

                candidate_pool = candidates
                if self.frame_summary_mode and layer_frame_ids is not None:
                    frame_ids = layer_frame_ids[layer_idx][
                        start_idx:start_idx + score.numel()
                    ].to(device=score.device, dtype=torch.long)
                    candidate_frames = frame_ids.index_select(0, candidates)
                    selected_visual_frames = frame_ids.index_select(
                        0, torch.nonzero(selected_mask, as_tuple=False).flatten()
                    )
                    selected_visual_frames = selected_visual_frames[
                        selected_visual_frames >= 0
                    ]
                    # Prefer tokens from frames that already have a selected
                    # token. Adding one there increases physical length by 1;
                    # adding the first token to a missing frame merely replaces
                    # its synthetic summary and leaves physical length flat.
                    candidate_pool = candidates[
                        torch.isin(candidate_frames, selected_visual_frames)
                    ]
                    if candidate_pool.numel() == 0:
                        candidate_pool = candidates

                take = min(int(needed), int(candidate_pool.numel()))
                candidate_scores = score.index_select(0, candidate_pool)
                extra_order = torch.topk(
                    candidate_scores, take, sorted=False
                ).indices
                extra_visual = candidate_pool.index_select(0, extra_order)
                selected_mask[extra_visual] = True
                keep = torch.cat((keep, extra_visual + start_idx))
                keep = torch.unique(keep, sorted=True)

            equalized.append(keep.tolist())

        return equalized

    def _build_frame_summary_specs(
        self,
        keep_indices_all_layers,
        layer_attention_scores,
        layer_configs,
        layer_selection_scores=None,
    ):
        """Describe one synthetic pooled token for each missing frame.

        The source indices point to the original, pre-pruning KV cache. The
        model-specific shrinker turns each source group into one K/V entry and
        assigns it the frame's mean attention score for tracing/debugging.
        """
        if not self.frame_summary_mode:
            return [[] for _ in keep_indices_all_layers]
        if self._token_frame_ids_per_layer is None:
            raise RuntimeError("Frame provenance is required for frame summaries")

        specs_per_layer = []
        for layer_idx, keep_indices in enumerate(keep_indices_all_layers):
            score = layer_attention_scores[layer_idx]
            selection_score = (
                layer_selection_scores[layer_idx]
                if layer_selection_scores is not None
                else score
            )
            start_idx = layer_configs[layer_idx]["visual_start_idx"]
            num_visual_tokens = score.numel()
            frame_ids = self._token_frame_ids_per_layer[layer_idx][
                start_idx:start_idx + num_visual_tokens
            ].to(device=score.device, dtype=torch.long)
            keep = torch.as_tensor(
                keep_indices, device=score.device, dtype=torch.long
            )
            selected_visual = keep[
                (keep >= start_idx) & (keep < start_idx + num_visual_tokens)
            ] - start_idx
            selected_mask = torch.zeros(
                num_visual_tokens, device=score.device, dtype=torch.bool
            )
            selected_mask[selected_visual] = True

            specs = []
            for frame_id in torch.unique(
                frame_ids[frame_ids >= 0], sorted=True
            ).tolist():
                frame_relative = torch.nonzero(
                    frame_ids == frame_id, as_tuple=False
                ).flatten()
                if selected_mask[frame_relative].any():
                    continue
                frame_score = score.index_select(0, frame_relative).mean()
                spec = {
                    "frame_id": int(frame_id),
                    "source_indices": (frame_relative + start_idx).detach().cpu(),
                    "attention_score": float(frame_score.item()),
                    "pool_strategy": self.frame_summary_strategy,
                }
                if self.frame_summary_strategy == "attention_weighted":
                    pool_weights = score.index_select(
                        0, frame_relative
                    ).float().clamp_min(0)
                    weight_sum = pool_weights.sum()
                    if not torch.isfinite(weight_sum) or weight_sum <= 0:
                        pool_weights = torch.full_like(
                            pool_weights, 1.0 / max(pool_weights.numel(), 1)
                        )
                    else:
                        pool_weights = pool_weights / weight_sum
                    spec["pool_weights"] = pool_weights.detach().cpu()
                elif self.frame_summary_strategy == "softmax_score_weighted":
                    pool_scores = selection_score.index_select(
                        0, frame_relative
                    ).float()
                    pool_weights = torch.softmax(
                        pool_scores / self.frame_summary_temperature, dim=0
                    )
                    spec["pool_weights"] = pool_weights.detach().cpu()
                specs.append(spec)
            specs_per_layer.append(specs)

        return specs_per_layer

    @staticmethod
    def _pool_frame_states(k_source, v_source, spec):
        """Pool aligned source K/V states according to a summary specification."""
        weights = spec.get("pool_weights")
        if weights is None:
            return (
                k_source.mean(dim=2, keepdim=True),
                v_source.mean(dim=2, keepdim=True),
            )

        weights = torch.as_tensor(
            weights, device=k_source.device, dtype=torch.float32
        )
        if weights.numel() != k_source.shape[2]:
            raise RuntimeError(
                "Frame-summary weights do not match source-token count: "
                f"{weights.numel()} != {k_source.shape[2]}"
            )
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        weights = weights.view(1, 1, -1, 1)
        pooled_k = (k_source.float() * weights).sum(dim=2, keepdim=True)
        pooled_v = (v_source.float() * weights).sum(dim=2, keepdim=True)
        return pooled_k.to(k_source.dtype), pooled_v.to(v_source.dtype)

    @staticmethod
    def _count_frame_tokens(frame_ids):
        counts = Counter(
            int(frame_id)
            for frame_id in frame_ids.tolist()
            if int(frame_id) >= 0
        )
        summary_tokens = int((frame_ids == -2).sum().item())
        return counts, summary_tokens

    def _record_token_retention(
        self,
        frame_ids_before,
        cache_lengths_before,
        compression_applied,
    ):
        """Record post-compression visual-token counts for every frame/layer."""
        if not self.token_trace_enabled:
            return
        if self._token_frame_ids_per_layer is None:
            return

        max_frame = self.total_processed_frames
        cache_lengths_after = self._get_cache_seq_len_per_layer()
        event_visual_before = 0
        event_visual_after = 0
        for layer_idx, ids_after in enumerate(self._token_frame_ids_per_layer):
            ids_before = frame_ids_before[layer_idx]
            before_counts, before_summaries = self._count_frame_tokens(ids_before)
            after_counts, after_summaries = self._count_frame_tokens(ids_after)
            summary_scores_after = None
            if self._token_frame_summary_scores_per_layer is not None:
                summary_scores_after = (
                    self._token_frame_summary_scores_per_layer[layer_idx]
                )
            visual_before = sum(before_counts.values())
            visual_after = sum(after_counts.values())
            event_visual_before += visual_before
            event_visual_after += visual_after
            average_before = visual_before / max_frame if max_frame else 0.0
            average_after = visual_after / max_frame if max_frame else 0.0

            if layer_idx < self.short_term_threshold:
                layer_type = "short-term"
            elif layer_idx >= self.long_term_threshold:
                layer_type = "long-term"
            else:
                layer_type = "mid-term"

            for frame_idx in range(max_frame):
                kept = after_counts.get(frame_idx, 0)
                before = before_counts.get(frame_idx, 0)
                frame_summary_count = 0
                frame_summary_score = ""
                if summary_scores_after is not None:
                    frame_scores = summary_scores_after[ids_after == frame_idx]
                    frame_scores = frame_scores[~torch.isnan(frame_scores)]
                    frame_summary_count = int(frame_scores.numel())
                    if frame_summary_count:
                        frame_summary_score = float(frame_scores.mean().item())
                self.token_trace_rows.append({
                    "video_id": self._token_trace_video_id,
                    "compression_event": self._token_trace_event,
                    "layer": layer_idx,
                    "layer_type": layer_type,
                    "frame_index": frame_idx,
                    "tokens_in_cache_before": before,
                    "tokens_kept_after": kept,
                    "tokens_dropped": before - kept,
                    "visual_tokens_before_layer": visual_before,
                    "visual_tokens_after_layer": visual_after,
                    "average_tokens_before_per_frame": average_before,
                    "average_tokens_after_per_frame": average_after,
                    "summary_tokens_before": before_summaries,
                    "summary_tokens_after": after_summaries,
                    "frame_summary_tokens_after": frame_summary_count,
                    "frame_summary_attention_score": frame_summary_score,
                    "cache_tokens_before_layer": cache_lengths_before[layer_idx],
                    "cache_tokens_after_layer": cache_lengths_after[layer_idx],
                    "compression_applied": int(compression_applied),
                })

        denominator = max(max_frame * len(self._token_frame_ids_per_layer), 1)
        if self.token_trace_verbose:
            print(
                f"Token retention video={self._token_trace_video_id} "
                f"event={self._token_trace_event}: average visual tokens/frame "
                f"before={event_visual_before / denominator:.2f}, "
                f"after={event_visual_after / denominator:.2f}"
            )
        self._token_trace_event += 1

    def write_token_trace(self):
        """Write the collected token-retention rows to the configured CSV."""
        if not self.token_trace_enabled:
            return
        path = Path(self.token_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video_id",
            "compression_event",
            "layer",
            "layer_type",
            "frame_index",
            "tokens_in_cache_before",
            "tokens_kept_after",
            "tokens_dropped",
            "visual_tokens_before_layer",
            "visual_tokens_after_layer",
            "average_tokens_before_per_frame",
            "average_tokens_after_per_frame",
            "summary_tokens_before",
            "summary_tokens_after",
            "frame_summary_tokens_after",
            "frame_summary_attention_score",
            "cache_tokens_before_layer",
            "cache_tokens_after_layer",
            "compression_applied",
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.token_trace_rows)

        summary_path = path.with_name(f"{path.stem}_summary.csv")
        summary_fields = [
            "scope",
            "video_id",
            "compression_event",
            "layer",
            "layer_type",
            "num_frames",
            "visual_tokens_before_layer",
            "visual_tokens_after_layer",
            "average_tokens_before_per_frame",
            "average_tokens_after_per_frame",
            "frame_summary_tokens_after",
            "frame_summary_attention_score",
            "cache_tokens_before_layer",
            "cache_tokens_after_layer",
            "compression_applied",
        ]
        summary_rows = []
        grouped_rows = {}
        grouped_counts = Counter()
        grouped_summary_counts = Counter()
        grouped_summary_score_sums = Counter()
        grouped_summary_score_counts = Counter()
        for row in self.token_trace_rows:
            key = (
                row["video_id"],
                row["compression_event"],
                row["layer"],
            )
            grouped_rows.setdefault(key, row)
            grouped_counts[key] += 1
            grouped_summary_counts[key] += int(
                row["frame_summary_tokens_after"] or 0
            )
            if row["frame_summary_attention_score"] != "":
                grouped_summary_score_sums[key] += float(
                    row["frame_summary_attention_score"]
                )
                grouped_summary_score_counts[key] += 1

        for key, row in grouped_rows.items():
            summary_rows.append({
                "scope": "event_layer",
                "video_id": row["video_id"],
                "compression_event": row["compression_event"],
                "layer": row["layer"],
                "layer_type": row["layer_type"],
                "num_frames": grouped_counts[key],
                "visual_tokens_before_layer": row["visual_tokens_before_layer"],
                "visual_tokens_after_layer": row["visual_tokens_after_layer"],
                "average_tokens_before_per_frame": row[
                    "average_tokens_before_per_frame"
                ],
                "average_tokens_after_per_frame": row[
                    "average_tokens_after_per_frame"
                ],
                "frame_summary_tokens_after": row[
                    "frame_summary_tokens_after"
                ],
                "frame_summary_attention_score": row[
                    "frame_summary_attention_score"
                ],
                "cache_tokens_before_layer": row["cache_tokens_before_layer"],
                "cache_tokens_after_layer": row["cache_tokens_after_layer"],
                "compression_applied": row["compression_applied"],
            })

            summary_rows[-1]["frame_summary_tokens_after"] = (
                grouped_summary_counts[key]
            )
            if grouped_summary_score_counts[key]:
                summary_rows[-1]["frame_summary_attention_score"] = (
                    grouped_summary_score_sums[key]
                    / grouped_summary_score_counts[key]
                )
            else:
                summary_rows[-1]["frame_summary_attention_score"] = ""

        if self.token_trace_rows:
            overall_average = sum(
                float(row["tokens_kept_after"])
                for row in self.token_trace_rows
            ) / len(self.token_trace_rows)
            overall_summary_count = sum(
                int(row["frame_summary_tokens_after"] or 0)
                for row in self.token_trace_rows
            )
            overall_summary_scores = [
                float(row["frame_summary_attention_score"])
                for row in self.token_trace_rows
                if row["frame_summary_attention_score"] != ""
            ]
            summary_rows.append({
                "scope": "overall",
                "average_tokens_after_per_frame": overall_average,
                "frame_summary_tokens_after": overall_summary_count,
                "frame_summary_attention_score": (
                    sum(overall_summary_scores) / len(overall_summary_scores)
                    if overall_summary_scores else ""
                ),
            })
            print(
                "Average retained visual tokens per frame/layer/event: "
                f"{overall_average:.2f}"
            )

        with summary_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Token-retention summary saved to: {summary_path}")

    def clear_cache(self):
        self.kv_cache = None
        self.conv_history = []
        self._reset_token_frame_ids()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        
        if mc:
            prompt += 'Best option: ('
        return prompt

    def get_gpu_memory_usage_gb(self):
        return torch.cuda.max_memory_allocated() / (1024**3)
