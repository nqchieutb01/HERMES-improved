"""Query-aware frame retrieval from a SigLIP 2 frame index (phase 23; index built by logs/phase23/embed_siglip.py).

The index holds one L2-normalised SigLIP 2 embedding per second of video, each frame embedded alone, and one text
embedding per question. Relevance of second j to a question asked at time t is the cosine similarity of their
embeddings, z-scored over the seconds 0..t only (causal: nothing after the question time is read).

Both uses mix relevance with uniform coverage, because grounding needs the whole window (logs/phase21/pruning_link.md):
    density_j = (1 - mix) / F + mix * softmax(z)_j          (mix = 0.5, softmax at temperature 1; fixed a priori)
- ``budget_shares``: per-frame token budget shares for the frames already in memory (R1, retrieval-weighted budget);
  each memory frame is credited with the density of the seconds it is nearest to.
- ``sample_times``: frame times drawn by inverse transform sampling of the density (R2, retrieval-guided frames,
  as in BOLT, Liu et al. CVPR 2025), at evenly spaced quantiles so the draw is deterministic.
  ``continuous=True`` treats the density as piecewise constant over each second and returns exactly ``num_frames``
  distinct times even when the window has fewer seconds than frames (the snapped version returns one frame per second).
- ``token_log_density``: the same mixture over the tokens of one frame (CoRe token sampling, logits of the density).
"""

from __future__ import annotations

import functools
import os

import numpy as np

INDEX = os.environ.get("HERMES_RETRIEVAL_INDEX", "/nfs-stor/chieu.nguyen/s-ember/siglip2_1fps")
MIX = 0.5


@functools.lru_cache(maxsize=4)
def _questions(kind):
    d = np.load(f"{INDEX}/questions_{kind}.npz")
    return {q: i for i, q in enumerate(d["question_id"].tolist())}, d["emb"].astype(np.float32)


@functools.lru_cache(maxsize=8)
def _video(video_id):
    d = np.load(f"{INDEX}/{video_id}.npz")
    return d["times"].astype(np.float64), d["emb"].astype(np.float32)


def relevance(video_id, question_id, end_time, kind="grounding"):
    """Seconds 0..end_time of the video and their z-scored similarity to the question."""
    index, q_emb = _questions(kind)
    times, emb = _video(video_id)
    keep = times <= float(end_time) + 1e-6
    times, emb = times[keep], emb[keep]
    sim = emb @ q_emb[index[question_id]]
    z = (sim - sim.mean()) / (sim.std() + 1e-6)
    return times, z


def density(z, mix=MIX):
    w = np.exp(z - z.max())
    return (1.0 - mix) / len(z) + mix * w / w.sum()


def budget_shares(frame_times, video_id, question_id, end_time, kind="grounding", mix=MIX):
    """Token-budget share of each memory frame (sums to 1): the density mass of the seconds nearest to it."""
    times, z = relevance(video_id, question_id, end_time, kind)
    p = density(z, mix)
    ft = np.asarray(frame_times, dtype=np.float64)
    nearest = np.abs(times[:, None] - ft[None, :]).argmin(axis=1)
    shares = np.bincount(nearest, weights=p, minlength=len(ft))
    shares = np.maximum(shares, 1e-9)
    return (shares / shares.sum()).tolist()


def sample_times(num_frames, video_id, question_id, end_time, kind="grounding", mix=MIX, continuous=False):
    """``num_frames`` frame times drawn at the quantiles (k + 0.5) / n of the density (sorted)."""
    times, z = relevance(video_id, question_id, end_time, kind)
    if continuous:
        p = density(z, mix)
        cdf = np.concatenate([[0.0], np.cumsum(p)])
        u = (np.arange(num_frames) + 0.5) / num_frames
        j = np.searchsorted(cdf, u, side="right").clip(1, len(times)) - 1
        frac = (u - cdf[j]) / np.maximum(p[j], 1e-12)
        span = np.minimum(1.0, float(end_time) - times[j]).clip(min=0.0)  # the last second may be partial
        return np.sort(times[j] + frac * span).tolist()
    if len(times) <= num_frames:
        return times.tolist()
    cdf = np.cumsum(density(z, mix))
    picks = np.searchsorted(cdf, (np.arange(num_frames) + 0.5) / num_frames).clip(0, len(times) - 1)
    chosen, used = [], set()
    for j in picks.tolist():
        # Dense peaks map several quantiles to one second: take the nearest unused second instead.
        for d in range(len(times)):
            cand = [c for c in (j - d, j + d) if 0 <= c < len(times) and c not in used]
            if cand:
                used.add(cand[0])
                chosen.append(cand[0])
                break
    return sorted(times[c] for c in chosen)


def token_log_density(scores, mix=MIX):
    """Log of the coverage-floored relevance density over the tokens of one frame (scores: any relevance, 1-D)."""
    import torch

    z = (scores - scores.mean()) / (scores.std() + 1e-6) if scores.numel() > 1 else torch.zeros_like(scores)
    w = torch.softmax(z, dim=0)
    return torch.log((1.0 - mix) / scores.numel() + mix * w)
