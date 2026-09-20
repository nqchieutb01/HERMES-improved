"""CPU regression checks against the pinned Transformers Qwen2 implementation.

Run: PYTHONPATH=. .venv/hermes/bin/python -m unittest tests.test_llava_attention
"""
import unittest
from types import MethodType, SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.abstract_hermes import Abstract_Hermes
from inference.llavaov_hermes import LlavaOneVision_Hermes
from inference.reindex_1d import (
    apply_rotary_delta_to_keys_only,
    apply_rotary_pos_emb_1d,
    compute_cos_sin_for_positions,
    rotary_delta,
)


class CompressionAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        config = Qwen2Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
            max_position_embeddings=256, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        self.model = Qwen2ForCausalLM(config).eval()
        with torch.inference_mode():
            self.cache = self.model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=True).past_key_values
        self.probe = SimpleNamespace(
            language_model=self.model,
            get_input_embeddings=self.model.get_input_embeddings,
            _get_next_start_pos_per_layer=lambda: [5, 5, 5],
            _build_position_ids=lambda start, length, batch: torch.arange(
                start, start + length).view(1, -1).expand(batch, -1),
        )

    def scores(self, ids):
        return LlavaOneVision_Hermes._compute_attention_scores_manually(
            self.probe, ids, self.cache)

    @torch.inference_mode()
    def test_matches_full_decoder_forward(self):
        ids = torch.tensor([[6, 7, 8, 9]])
        expected = self.model(ids, past_key_values=self.cache, use_cache=True,
                              output_attentions=True).attentions
        actual = self.scores(ids)
        for layer, (observed, reference) in enumerate(zip(actual, expected)):
            with self.subTest(layer=layer):
                torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-7)

    @torch.inference_mode()
    def test_future_query_tokens_cannot_change_earlier_attention(self):
        before = self.scores(torch.tensor([[6, 7, 8, 9]]))
        after = self.scores(torch.tensor([[6, 7, 10, 11]]))
        for observed, reference in zip(before, after):
            torch.testing.assert_close(observed[:, :, :2], reference[:, :, :2])

    @torch.inference_mode()
    def test_scoring_does_not_mutate_visual_cache(self):
        original = [(k.clone(), v.clone()) for k, v in self.cache]
        self.scores(torch.tensor([[6, 7, 8]]))
        for (k, v), (old_k, old_v) in zip(self.cache, original):
            torch.testing.assert_close(k, old_k, rtol=0, atol=0)
            torch.testing.assert_close(v, old_v, rtol=0, atol=0)

    @torch.inference_mode()
    def test_matches_forward_with_layer_specific_logical_positions(self):
        # Streaming compression leaves gaps in RoPE positions, with different
        # next positions across layers even when physical cache lengths match.
        starts = [15, 25, 35]
        ids = torch.tensor([[6, 7, 8]])
        self.probe._get_next_start_pos_per_layer = lambda: starts
        LlavaOneVision_Hermes._patch_rotary_embeddings(self.probe)
        LlavaOneVision_Hermes._set_rotary_required_len(self.probe, 38)
        handles = []
        for layer, start in zip(self.model.model.layers, starts):
            def hook(module, args, kwargs, start=start):
                kwargs['position_ids'] = torch.arange(start, start + 3).view(1, -1)
                return args, kwargs
            handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            expected = self.model(ids, past_key_values=self.cache, use_cache=True,
                                  output_attentions=True).attentions
        finally:
            for handle in handles:
                handle.remove()
        for observed, reference in zip(self.scores(ids), expected):
            torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-7)


class FrameFloorCompressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        config = Qwen2Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
            max_position_embeddings=256, attention_dropout=0.0,
        )
        self.model = Qwen2ForCausalLM(config).eval()

    @torch.inference_mode()
    def test_streaming_frame_summary_is_rerotated_before_pooling(self):
        """A synthetic token must have the RoPE phase of its assigned position."""
        positions = torch.arange(5, dtype=torch.long)
        base_keys = torch.randn(1, 2, 5, 8)
        cos, sin = compute_cos_sin_for_positions(
            self.model, positions.numel(), positions, torch.float32,
            torch.device("cpu"),
        )
        rotated_keys = apply_rotary_pos_emb_1d(base_keys, cos, sin)
        values = torch.randn_like(rotated_keys)

        probe = SimpleNamespace(
            device=torch.device("cpu"),
            language_model=self.model,
            kv_cache=[(rotated_keys.clone(), values.clone())],
            num_layers=1,
            streaming=True,
            reindex_margin=64,
            _position_ids_cache=[positions.clone()],
            token_provenance_enabled=True,
            _token_frame_ids_per_layer=[torch.tensor([-1, 10, 10, 11, 11])],
            _token_frame_summary_scores_per_layer=[
                torch.full((5,), float("nan"))
            ],
            _frame_summary_specs_per_layer=[[{
                "frame_id": 10,
                "source_indices": torch.tensor([1, 2]),
                "attention_score": 0.25,
            }]],
            _pool_frame_states=Abstract_Hermes._pool_frame_states,
            token_trace_verbose=False,
            # Keep this layer out of the separate long-term fold path.
            long_term_threshold=2,
        )
        probe._get_cache_seq_len_per_layer = lambda: [
            probe.kv_cache[0][0].shape[2]
        ]
        probe._sanitize_keep_indices = MethodType(
            LlavaOneVision_Hermes._sanitize_keep_indices, probe
        )

        # Streaming mode does not compact at these small positions. The new
        # frame-summary token is appended at logical position 5.
        LlavaOneVision_Hermes._shrink_positions_and_rerotate_keys(
            probe, [[0, 3, 4]]
        )

        old_source_positions = positions[torch.tensor([1, 2])]
        target_positions = torch.full_like(old_source_positions, 5)
        cos_old, sin_old = compute_cos_sin_for_positions(
            self.model, 2, old_source_positions, torch.float32,
            torch.device("cpu"),
        )
        cos_new, sin_new = compute_cos_sin_for_positions(
            self.model, 2, target_positions, torch.float32,
            torch.device("cpu"),
        )
        cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)
        expected = apply_rotary_delta_to_keys_only(
            rotated_keys[:, :, [1, 2], :], cos_delta, sin_delta
        ).mean(dim=2)

        observed = probe.kv_cache[0][0][:, :, 3, :]
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(probe._position_ids_cache[0].tolist(), [0, 3, 4, 5])

    def test_top_patch_keeps_best_real_token_from_missing_frame(self):
        """top_patch must add the best real patch instead of a synthetic token."""
        probe = SimpleNamespace(
            min_tokens_per_frame=1,
            frame_summary_mode=False,
        )
        scores = torch.tensor([0.9, 0.1, 0.8, 0.7])
        frame_ids = torch.tensor([10, 10, 11, 11])

        selected, effective_budget = (
            Abstract_Hermes._select_indices_with_frame_minimum(
                probe, scores, frame_ids, budget=1
            )
        )

        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual(effective_budget, 2)

    def test_top_attention_patch_uses_raw_attention_for_frame_top_up(self):
        """The raw-attention variant must not reuse the blended HERMES score."""
        probe = SimpleNamespace(
            min_tokens_per_frame=1,
            frame_summary_mode=False,
        )
        selection_scores = torch.tensor([0.9, 0.1, 0.8, 0.7])
        attention_scores = torch.tensor([0.4, 0.3, 0.1, 0.9])
        frame_ids = torch.tensor([10, 10, 11, 11])

        selected, effective_budget = (
            Abstract_Hermes._select_indices_with_frame_minimum(
                probe,
                selection_scores,
                frame_ids,
                budget=1,
                frame_floor_scores=attention_scores,
            )
        )

        self.assertEqual(selected.tolist(), [0, 3])
        self.assertEqual(effective_budget, 2)

    def test_top_attention_patch_keeps_two_distinct_patches_per_frame(self):
        """k=2 must top each frame up with distinct raw-attention patches."""
        probe = SimpleNamespace(
            min_tokens_per_frame=2,
            frame_summary_mode=False,
        )
        selection_scores = torch.tensor([0.9, 0.1, 0.2, 0.8, 0.3, 0.4])
        attention_scores = torch.tensor([0.1, 0.2, 0.9, 0.1, 0.2, 0.8])
        frame_ids = torch.tensor([10, 10, 10, 11, 11, 11])

        selected, effective_budget = (
            Abstract_Hermes._select_indices_with_frame_minimum(
                probe,
                selection_scores,
                frame_ids,
                budget=2,
                frame_floor_scores=attention_scores,
            )
        )

        self.assertEqual(selected.tolist(), [0, 2, 3, 5])
        self.assertEqual(effective_budget, 4)

    def test_attention_weighted_summary_uses_raw_attention(self):
        """Attention pooling must normalize weights inside the missing frame."""
        probe = SimpleNamespace(
            frame_summary_mode=True,
            frame_summary_strategy="attention_weighted",
            frame_summary_temperature=0.1,
            _token_frame_ids_per_layer=[
                torch.tensor([-1, 10, 10, 11, 11])
            ],
        )
        specs = Abstract_Hermes._build_frame_summary_specs(
            probe,
            keep_indices_all_layers=[[0, 3]],
            layer_attention_scores=[torch.tensor([1.0, 3.0, 2.0, 4.0])],
            layer_configs=[{"visual_start_idx": 1}],
            layer_selection_scores=[torch.tensor([0.8, 0.2, 0.7, 0.1])],
        )

        self.assertEqual(len(specs[0]), 1)
        self.assertEqual(specs[0][0]["source_indices"].tolist(), [1, 2])
        torch.testing.assert_close(
            specs[0][0]["pool_weights"], torch.tensor([0.25, 0.75])
        )

    def test_softmax_score_summary_and_pooling(self):
        """Softmax pooling must use final HERMES scores and pool in FP32."""
        probe = SimpleNamespace(
            frame_summary_mode=True,
            frame_summary_strategy="softmax_score_weighted",
            frame_summary_temperature=0.1,
            _token_frame_ids_per_layer=[
                torch.tensor([-1, 10, 10, 11, 11])
            ],
        )
        specs = Abstract_Hermes._build_frame_summary_specs(
            probe,
            keep_indices_all_layers=[[0, 3]],
            layer_attention_scores=[torch.tensor([1.0, 3.0, 2.0, 4.0])],
            layer_configs=[{"visual_start_idx": 1}],
            layer_selection_scores=[torch.tensor([0.8, 0.2, 0.7, 0.1])],
        )
        spec = specs[0][0]
        expected_weights = torch.softmax(torch.tensor([0.8, 0.2]) / 0.1, dim=0)
        torch.testing.assert_close(spec["pool_weights"], expected_weights)

        keys = torch.tensor([[[[2.0], [10.0]]]], dtype=torch.float16)
        values = torch.tensor([[[[4.0], [20.0]]]], dtype=torch.float16)
        pooled_k, pooled_v = Abstract_Hermes._pool_frame_states(
            keys, values, spec
        )
        torch.testing.assert_close(
            pooled_k.float().flatten(),
            torch.tensor([2.0]) * expected_weights[0]
            + torch.tensor([10.0]) * expected_weights[1],
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            pooled_v.float().flatten(),
            torch.tensor([4.0]) * expected_weights[0]
            + torch.tensor([20.0]) * expected_weights[1],
            rtol=1e-3,
            atol=1e-3,
        )

    @torch.inference_mode()
    def test_long_term_selection_reserves_space_for_fold_summary(self):
        """The configured visual budget includes the long-term fold token."""
        cache_length = 8
        visual_start = 2
        attention = torch.rand(1, 1, 1, cache_length + 1)
        probe = SimpleNamespace(
            device=torch.device("cpu"),
            visual_start_idx=visual_start,
            token_activity_cache=[None],
            short_term_threshold=0,
            long_term_threshold=0,
            recency_weight_start=0.75,
            recency_weight_decay=0.6,
            min_tokens_per_frame=0,
            frame_summary_mode=False,
            _token_frame_ids_per_layer=None,
        )
        probe._get_cache_seq_len_per_layer = lambda: [cache_length]
        probe.allocate_budget_by_depth = MethodType(
            LlavaOneVision_Hermes.allocate_budget_by_depth, probe
        )
        probe._select_indices_with_frame_minimum = MethodType(
            Abstract_Hermes._select_indices_with_frame_minimum, probe
        )
        probe._equalize_keep_indices_by_layer = MethodType(
            Abstract_Hermes._equalize_keep_indices_by_layer, probe
        )
        probe._build_frame_summary_specs = MethodType(
            Abstract_Hermes._build_frame_summary_specs, probe
        )

        keep = LlavaOneVision_Hermes.prune_kv_cache_by_attention(
            probe, [attention], [attention], [attention], num_keep=4
        )

        # Two prompt tokens + three selected visual tokens + one fold token
        # equals the requested total post-compression budget of 2 + 4.
        self.assertEqual(len(keep[0]), visual_start + 3)


if __name__ == "__main__":
    unittest.main()
