"""VidCom2 (Liu et al., EMNLP 2025): frame-uniqueness budgets + outlier token selection, before the LLM.

Training-free, query-agnostic and attention-free (feature statistics only). This is a line-by-line port of the authors'
released code (github.com/xuyang-liu16/VidCom2, token_compressor/vidcom2/vidcom2.py and models/qwen3_vl.py), which
differs from the paper's equations: scores use the lowest-variance half of the channels and a multi-scale Gaussian
similarity to the video and frame centres instead of cosine similarity. As in the authors' Qwen3-VL integration, a
"frame" is one temporal group (two source frames) and DeepStack features are kept at the same indices.

Inputs per video: ``features`` [G, N, D] (merged visual tokens), ``retention`` r (the authors' ``base_scale``).
Output: for each group, the kept token indices (sorted) and a function selecting them from any aligned tensor [G, N, ...].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _select_low_var_channels(x, ratio=0.5):
    variances = x.var(dim=0, unbiased=False)
    _, idx = torch.topk(variances, k=int(x.shape[-1] * ratio), largest=False)
    return x[:, idx]


def _multi_scale_gaussian(x, center, alphas):
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * a)) for a in alphas)


@torch.no_grad()
def vidcom2(features, retention, temp=0.01):
    num_groups, tpf, dim = features.shape
    sel = _select_low_var_channels(features.reshape(num_groups * tpf, dim))
    frames = F.normalize(sel.view(num_groups, tpf, -1), dim=-1)
    alphas = [2 ** k for k in range(-3, 2)]
    vid_score = _multi_scale_gaussian(frames, frames.mean(dim=(0, 1), keepdim=True), alphas)
    frame_score = _multi_scale_gaussian(frames, frames.mean(dim=1, keepdim=True), alphas)
    # Frame budgets: more tokens for frames less similar to the whole video.
    scores = -vid_score.mean(dim=-1)
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    scales = (retention * (1 + probs - probs.mean())).clamp(max=1.0)
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    # Within each frame keep the tokens least similar to both the video and the frame centre.
    total = vid_score + frame_score
    keep = [torch.topk(total[g], k=k, largest=False, sorted=False).indices.sort().values for g, k in enumerate(ks)]

    def apply(tensor):
        return [tensor[g][keep[g]] for g in range(num_groups)]

    return [k.cpu() for k in keep], apply
