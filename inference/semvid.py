"""SemVID (Li et al., ECCV 2026): semantic evidence allocation for training-free token pruning in temporal grounding.

Query-aware selection on the vision encoder's output before the LLM. This is a line-by-line port of the authors'
``_semantic_prune_video_region`` (github.com/JiaqiLi404/SemVID, src/models/models/modeling_qwen3_vl_semvid.py) with
their released Qwen3-VL settings (configs/_base_/methods/semvid_qwen3.py): no DySeg segmentation (c = tau = 0, one
segment), 1 token floor per frame, frame weight 0.7 * relu(query relevance) + 0.3 * motion energy, 60% object tokens
by per-query-token MMR (lambda 0.9), 40% motion tokens (temporal difference blended 0.5 with query similarity), the
rest by token norm, one context prototype per frame. A "frame" is one temporal group of Qwen3-VL, as in the original.

Query: the question's token embeddings from the LLM input embedding table (``_compute_query_embedding``: per-token,
L2-normalised, special tokens excluded). Token saliency: the norm of the merged visual token (their
``return_token_scores``). Deviation: the authors drop Qwen3-VL's DeepStack features after pruning; here DeepStack is
kept at the kept indices, as for every other selector in this repository.

Inputs: ``features`` [G, N, D], ``query_tokens`` [L, D] (raw embeddings), ``retention`` r.
Output: kept token indices per group (sorted) and a function selecting them from any aligned tensor [G, N, ...].
"""

from __future__ import annotations

import torch

DEFAULTS = dict(base_k=1, alpha=0.7, obj_ratio=0.6, mmr_lambda=0.9, motion_query_beta=0.5, tau=0.0)


def _allocate(weights, total):
    if total <= 0:
        return torch.zeros_like(weights, dtype=torch.long)
    w = weights.float().clamp_min(0)
    if w.sum() < 1e-8:
        w = torch.ones_like(w)
    raw = w / w.sum() * float(total)
    flo = torch.floor(raw).to(torch.long)
    rem = int(total - flo.sum().item())
    if rem > 0:
        flo[torch.topk(raw - flo.float(), k=min(rem, raw.numel())).indices] += 1
    return flo


def _allocate_with_cap(weights, total, cap):
    total = int(min(total, int(cap.sum().item())))
    alloc = torch.zeros_like(cap, dtype=torch.long)
    if total <= 0:
        return alloc
    active, remaining = cap > 0, total
    for _ in range(16):
        if remaining <= 0 or not bool(active.any()):
            break
        w = weights.float().clamp_min(0) * active.float()
        if w.sum() < 1e-8:
            w = active.float()
        raw = w / w.sum() * float(remaining)
        flo = torch.floor(raw).to(torch.long)
        rem = int(remaining - flo.sum().item())
        if rem > 0:
            frac = (raw - flo.float()).masked_fill(~active, -1.0)
            flo[torch.topk(frac, k=min(rem, int(active.sum().item()))).indices] += 1
        flo = torch.minimum(flo, cap - alloc)
        used = int(flo.sum().item())
        alloc, remaining = alloc + flo, remaining - used
        active = (cap - alloc) > 0
        if used == 0:
            break
    return alloc


def _mmr(rel, emb, k, lam):
    n = int(rel.numel())
    if k <= 0 or n <= 0:
        return rel.new_empty((0,), dtype=torch.long)
    k = min(int(k), n)
    first = int(torch.argmax(rel).item())
    selected = [first]
    available = torch.ones(n, device=rel.device, dtype=torch.bool)
    available[first] = False
    redundancy = emb @ emb[first]
    for _ in range(k - 1):
        if not bool(available.any()):
            break
        nxt = int(torch.argmax((lam * rel - (1.0 - lam) * redundancy).masked_fill(~available, -1e9)).item())
        selected.append(nxt)
        available[nxt] = False
        redundancy = torch.maximum(redundancy, emb @ emb[nxt])
    return torch.tensor(selected, device=rel.device, dtype=torch.long)


@torch.no_grad()
def semvid(features, query_tokens, retention, base_k=1, alpha=0.7, obj_ratio=0.6, mmr_lambda=0.9,
           motion_query_beta=0.5, tau=0.0):
    T, P, D = features.shape
    device = features.device
    feats = features.float()
    fg = feats.mean(dim=1)
    fg_norm = fg / fg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q_tokens = query_tokens.float().to(device)
    q_tokens = q_tokens / q_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rel_mat = fg_norm @ q_tokens.t()
    frame_rel = rel_mat.squeeze(1) if rel_mat.shape[1] == 1 else torch.topk(rel_mat, k=min(2, rel_mat.shape[1]),
                                                                             dim=1).values.mean(dim=1)
    base_k = max(min(base_k, P), 1)
    target = min(max(int(round(T * P * retention)), T * base_k), T * P)
    # DySeg with the released c = tau = 0: cuts only where adjacent frames have negative cosine similarity.
    seg_sizes = [T]
    if T > 1:
        cut = torch.nonzero((fg_norm[:-1] * fg_norm[1:]).sum(dim=-1) < tau).squeeze(1).sort().values
        if cut.numel() > 0:
            seg_sizes = [int(cut[0].item()) + 1] + [int(cut[i].item() - cut[i - 1].item()) for i in range(1, cut.numel())]
            seg_sizes = [x for x in seg_sizes + [int(T - cut[-1].item() - 1)] if x > 0]
    # Stage 1: the extra budget goes to segments by relevance, then to frames by 0.7 * relevance + 0.3 * motion.
    motion_energy = torch.zeros(T, device=device)
    if T > 1:
        motion_energy[1:] = (fg_norm[1:] - fg_norm[:-1]).norm(dim=-1)
        motion_energy[0] = motion_energy[1]
    frame_keep = torch.full((T,), base_k, device=device, dtype=torch.long)
    remaining = target - T * base_k
    if remaining > 0:
        starts = [sum(seg_sizes[:i]) for i in range(len(seg_sizes))]
        seg_scores = torch.stack([frame_rel[st:st + sz].mean() for st, sz in zip(starts, seg_sizes)])
        w_seg = torch.relu(seg_scores)
        if w_seg.sum() < 1e-8:
            w_seg = (seg_scores - seg_scores.min()).clamp_min(0) + 1e-6
        seg_extra = _allocate(w_seg, remaining)
        for i, (st, sz) in enumerate(zip(starts, seg_sizes)):
            extra = int(seg_extra[i].item())
            if extra > 0:
                w_frm = alpha * torch.relu(frame_rel[st:st + sz]) + (1.0 - alpha) * motion_energy[st:st + sz].clamp_min(0)
                if w_frm.sum() < 1e-8:
                    w_frm = torch.ones_like(w_frm)
                frame_keep[st:st + sz] += _allocate(w_frm, extra)
        frame_keep = frame_keep.clamp(max=P)
        if int(frame_keep.sum().item()) < target:
            slack = (P - frame_keep).clamp_min(0)
            w_all = (alpha * torch.relu(frame_rel) + (1.0 - alpha) * motion_energy.clamp_min(0)) * slack.float()
            if w_all.sum() < 1e-8:
                w_all = slack.float()
            frame_keep += _allocate_with_cap(w_all, target - int(frame_keep.sum().item()), slack)
    # Stage 2: context prototype, query objects (per-query-token MMR), motion, then fill by token norm.
    scores = feats.norm(dim=-1)
    keep = []
    for t in range(T):
        k_total = min(int(frame_keep[t].item()), P)
        sel = torch.zeros(P, device=device, dtype=torch.bool)
        if k_total <= 0:
            keep.append(torch.nonzero(sel).flatten())
            continue
        tok_norm = feats[t] / feats[t].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim_tok_q = tok_norm @ q_tokens.t()
        sel[int(torch.argmax(tok_norm @ fg_norm[t]).item())] = True
        extra = k_total - int(sel.sum().item())
        if extra > 0:
            k_obj = max(min(int(round(extra * obj_ratio)), extra), 0)
            k_mot = extra - k_obj
            if k_obj > 0:
                w_q = sim_tok_q.max(dim=0).values.clamp_min(0)
                alloc_q = _allocate(w_q, k_obj)
                for qi in torch.argsort(w_q, descending=True).tolist():
                    ki = int(alloc_q[qi].item())
                    if ki <= 0:
                        continue
                    cand = torch.arange(P, device=device)[~sel]
                    if cand.numel() == 0:
                        break
                    picked = _mmr(sim_tok_q[cand, qi], tok_norm[cand], ki, mmr_lambda)
                    sel[cand[picked]] = True
            if k_mot > 0:
                cand = torch.arange(P, device=device)[~sel]
                if cand.numel() > 0:
                    if T == 1:
                        mot = torch.zeros(P, device=device)
                    elif t == 0:
                        mot = (feats[1] - feats[0]).norm(dim=-1)
                    elif t == T - 1:
                        mot = (feats[t] - feats[t - 1]).norm(dim=-1)
                    else:
                        mot = 0.5 * ((feats[t] - feats[t - 1]).norm(dim=-1) + (feats[t + 1] - feats[t]).norm(dim=-1))
                    ms = mot[cand]
                    if motion_query_beta > 1e-6:
                        sim_pos = sim_tok_q[cand].max(dim=1).values.clamp_min(0)
                        ms = (1.0 - motion_query_beta) * ms / ms.max().clamp_min(1e-6) + \
                            motion_query_beta * sim_pos / sim_pos.max().clamp_min(1e-6)
                    k_mot = min(k_mot, int(cand.numel()))
                    sel[cand[torch.topk(ms, k=k_mot).indices]] = True
            need = k_total - int(sel.sum().item())
            if need > 0:
                cand = torch.arange(P, device=device)[~sel]
                if cand.numel() > 0:
                    sel[cand[torch.topk(scores[t][cand], k=min(need, int(cand.numel()))).indices]] = True
        keep.append(torch.nonzero(sel).flatten())

    def apply(tensor):
        return [tensor[g][keep[g]] for g in range(T)]

    return [k.cpu() for k in keep], apply
