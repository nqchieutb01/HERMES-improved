"""Rotary helpers for Qwen3-VL's interleaved M-RoPE cache entries."""

from __future__ import annotations

import torch


def _get_rotary_module(language_model):
    rotary = getattr(language_model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("Qwen3-VL language model has no rotary_emb module")
    return rotary


def compute_cos_sin_for_positions(
    language_model,
    seq_len: int,
    position_ids_3d: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
):
    """Compute Qwen3-VL's already-interleaved M-RoPE tensors."""
    if position_ids_3d.dim() == 2:
        position_ids_3d = position_ids_3d.unsqueeze(1)
    position_ids_3d = position_ids_3d.to(device)
    hidden_size = int(getattr(language_model.config, "hidden_size", 4096))
    dummy_hidden = torch.zeros(
        (1, seq_len, hidden_size), device=device, dtype=dtype
    )
    cos, sin = _get_rotary_module(language_model)(
        dummy_hidden, position_ids_3d
    )
    return cos.to(dtype), sin.to(dtype)


def rotary_delta(cos_old, sin_old, cos_new, sin_new):
    """Return the rotation that maps an old RoPE phase to a new phase."""
    cos_delta = cos_new * cos_old + sin_new * sin_old
    sin_delta = sin_new * cos_old - cos_new * sin_old
    return cos_delta, sin_delta


def rotate_half(states: torch.Tensor) -> torch.Tensor:
    """Match Transformers' Qwen3-VL ``rotate_half`` implementation."""
    first, second = states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply already-interleaved Qwen3-VL rotary tensors to Q and K."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_embed = query_states * cos + rotate_half(query_states) * sin
    key_embed = key_states * cos + rotate_half(key_states) * sin
    return query_embed, key_embed


def apply_rotary_delta_to_keys_only(
    key_states: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
) -> torch.Tensor:
    """Move cached Qwen3-VL keys between logical positions."""
    _, key_embed = apply_rotary_pos_emb(
        key_states,
        key_states,
        cos_delta,
        sin_delta,
    )
    return key_embed


__all__ = [
    "apply_rotary_delta_to_keys_only",
    "apply_rotary_pos_emb",
    "compute_cos_sin_for_positions",
    "rotary_delta",
]
