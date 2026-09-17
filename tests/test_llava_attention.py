"""CPU regression checks against the pinned Transformers Qwen2 implementation.

Run: PYTHONPATH=. .venv/hermes/bin/python -m unittest tests.test_llava_attention
"""
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.llavaov_hermes import LlavaOneVision_Hermes


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


if __name__ == "__main__":
    unittest.main()
