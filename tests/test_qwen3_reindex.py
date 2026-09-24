"""CPU checks for Qwen3-VL interleaved M-RoPE cache movement."""

import unittest

import torch

from inference.reindex_qwen3 import (
    apply_rotary_delta_to_keys_only,
    apply_rotary_pos_emb,
    rotary_delta,
)
from inference.qwen3vl_hermes import get_attention_sequence_ids

# Importing this backend installs the compatibility adapter used by Qwen3-VL
# when Transformers prepares packed FlashAttention sequences.
import inference.qwenvl_hermes  # noqa: F401, E402
from transformers.modeling_flash_attention_utils import (
    prepare_fa_kwargs_from_position_ids,
)


class Qwen3RotaryTest(unittest.TestCase):
    def test_rotary_delta_matches_direct_target_rotation(self):
        torch.manual_seed(2024)
        base_keys = torch.randn(1, 2, 4, 8)
        old_angles = torch.randn(1, 4, 4)
        new_angles = torch.randn(1, 4, 4)
        old_angles = torch.cat((old_angles, old_angles), dim=-1)
        new_angles = torch.cat((new_angles, new_angles), dim=-1)
        cos_old, sin_old = old_angles.cos(), old_angles.sin()
        cos_new, sin_new = new_angles.cos(), new_angles.sin()

        _, keys_at_old_positions = apply_rotary_pos_emb(
            base_keys, base_keys, cos_old, sin_old
        )
        _, expected = apply_rotary_pos_emb(
            base_keys, base_keys, cos_new, sin_new
        )
        cos_move, sin_move = rotary_delta(
            cos_old, sin_old, cos_new, sin_new
        )
        observed = apply_rotary_delta_to_keys_only(
            keys_at_old_positions, cos_move, sin_move
        )

        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


class QwenFlashAttentionCompatibilityTest(unittest.TestCase):
    def test_transformers_2d_signature_is_preserved(self):
        position_ids = torch.arange(4).unsqueeze(0)

        (cu_q, cu_k), (max_q, max_k) = (
            prepare_fa_kwargs_from_position_ids(position_ids)
        )

        torch.testing.assert_close(cu_q, torch.tensor([0, 4], dtype=torch.int32))
        torch.testing.assert_close(cu_k, cu_q)
        self.assertEqual((max_q, max_k), (4, 4))

    def test_qwen_3d_positions_are_treated_as_one_sequence_per_batch(self):
        position_ids = torch.zeros(3, 2, 5, dtype=torch.long)

        (cu_q, cu_k), (max_q, max_k) = (
            prepare_fa_kwargs_from_position_ids(position_ids)
        )

        expected = torch.tensor([0, 5, 10], dtype=torch.int32)
        torch.testing.assert_close(cu_q, expected)
        torch.testing.assert_close(cu_k, expected)
        self.assertEqual((max_q, max_k), (5, 5))

    def test_mrope_coordinates_use_monotonic_attention_sequence_ids(self):
        position_ids = torch.tensor(
            [
                [[14, 14, 15, 15], [20, 20, 21, 21]],
                [[14, 15, 14, 15], [20, 21, 20, 21]],
                [[14, 14, 15, 15], [20, 20, 21, 21]],
            ]
        )

        observed = get_attention_sequence_ids(position_ids)

        expected = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
