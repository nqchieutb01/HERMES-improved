"""TimeLens-Bench runs only (on PYTHONPATH via hermes_pool/timelens/anchor_chunk.sh).

TimeLens pins torch 2.6; our environment has torch 2.9, whose bf16 Conv3d is pathologically slow for Qwen3-VL's
patch embedding (~10 min per 300-frame video). The Conv3d there has kernel == stride over pre-flattened patches, so
it equals a linear layer with the same weights (as inference/qwenvl_hermes.py use_linear_patch_embed). Replace only
that forward; TimeLens' code and every other module are unchanged.
"""
try:
    import torch.nn.functional as F
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _m

    def _linear_patch_embed_forward(self, hidden_states):
        proj = self.proj
        weight = proj.weight.view(proj.out_channels, -1)
        hidden_states = hidden_states.view(-1, weight.shape[1]).to(dtype=weight.dtype)
        return F.linear(hidden_states, weight, proj.bias).view(-1, self.embed_dim)

    _m.Qwen3VLVisionPatchEmbed.forward = _linear_patch_embed_forward
except Exception as exc:  # never break interpreter start-up
    import sys
    print(f"sitecustomize: linear patch embed not applied: {exc}", file=sys.stderr)
