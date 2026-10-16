"""FastVID (Shen et al., NeurIPS 2025): dynamic temporal segmentation + density spatiotemporal pruning.

Training-free, query-agnostic visual token pruning applied to the vision encoder's output before the LLM. This is a
line-by-line port of the authors' Qwen2.5-VL implementation
(github.com/LunarShen/FastVID, fastvid_qwen25vl/.../modeling_qwen2_5_vl.py, ``forward`` after the vision encoder)
to per-group tensors; a "frame" here is one temporal group of Qwen3-VL (two source frames), as in the original.

Defaults are the authors' Qwen2.5-VL settings (scripts/eval.sh): c=8, tau=0.84, ignore=0.95, d=0.4, p=4, beta=0.6;
k=4 neighbours for the density (hard-coded in their code).

Inputs per video: ``features`` [G, N, D] (merged visual tokens, the LLM's input embeddings), ``attn`` [G, N] (salience
from the last vision block: mean frame query attending to keys pooled per merged token), ``retention`` r.
Output: for each group, the kept token indices (sorted) and a function that applies the same selection and
anchor-centric merging to any aligned tensor [G, N, ...] (used for the features and for Qwen3-VL's DeepStack features).
"""

from __future__ import annotations

import torch

DEFAULTS = dict(c=8, tau=0.84, ignore=0.95, d=0.4, p=4, beta=0.6, k=4)


def _sizes(cut_indices, total):
    sizes = [cut_indices[0] + 1]
    for i in range(1, len(cut_indices)):
        sizes.append(cut_indices[i] - cut_indices[i - 1])
    sizes.append(total - cut_indices[-1] - 1)
    return sizes


@torch.no_grad()
def fastvid(features, attn, retention, c=8, tau=0.84, ignore=0.95, d=0.4, p=4, beta=0.6, k=4, seed=0):
    num_frames, frm_len, _ = features.shape
    device = features.device
    gen = torch.Generator(device=device).manual_seed(seed)
    glob = features.float().mean(1)
    glob = glob / glob.norm(dim=1, keepdim=True)

    # Dynamic temporal segmentation: the c-1 least similar transitions and every transition below tau.
    if num_frames > 1:
        sim = (glob[:-1] * glob[1:]).sum(dim=1)
        cut = torch.topk(sim, min(c - 1, sim.numel()), largest=False).indices
        cut = torch.unique(torch.cat([cut, torch.nonzero(sim < tau).squeeze(1)])).sort().values.tolist()
        segment_sizes = _sizes(cut, num_frames) if cut else [num_frames]
    else:
        segment_sizes = [1]

    frame_retain_num = int(frm_len * retention)
    plan = []  # per segment: (first frame, length, salient idx, [(context idx, merge idx, one-hot, weights)])
    start = 0
    for cur_seg_len in segment_sizes:
        cur_glob = glob[start:start + cur_seg_len]
        cur_attn = attn[start:start + cur_seg_len]
        cur_feat = features[start:start + cur_seg_len]
        seg_retain_num = frame_retain_num * cur_seg_len
        seg_context_num = max(int(seg_retain_num * d), 1)
        seg_salient_num = seg_retain_num - seg_context_num

        if cur_seg_len == 1:
            frm_salient_num, frm_context_num = [seg_salient_num], [seg_context_num]
        else:
            # Near-duplicate runs inside the segment: tokens are taken from the last frame of each run.
            s = (cur_glob[:-1] * cur_glob[1:]).sum(dim=1)
            sub_cut = torch.nonzero(s < ignore).squeeze(1).sort().values.tolist()
            cur_segment_sizes = _sizes(sub_cut, cur_seg_len) if sub_cut else [cur_seg_len]
            valid_seg_len = len(cur_segment_sizes)
            chunk, rem = divmod(seg_salient_num, valid_seg_len)
            frm_salient = [chunk + (1 if i < rem else 0) for i in range(valid_seg_len)]
            temp_num = (valid_seg_len + p - 1) // p
            chunk, rem = divmod(seg_context_num, temp_num)
            temp_context = [chunk + (1 if i < rem else 0) for i in range(temp_num)]
            ctx_i, frm_context = 0, []
            for i in range(valid_seg_len):
                if i % p == 0:
                    frm_context.append(min(temp_context[ctx_i], frm_len // 2))
                    if temp_context[ctx_i] + frm_context[-1] > frm_len:
                        temp_context[ctx_i] = frm_len - frm_context[-1]
                    ctx_i += 1
                else:
                    frm_context.append(0)
            frm_context.reverse()
            frm_salient_num, frm_context_num = [], []
            for i in range(valid_seg_len):
                frm_salient_num.extend([0] * (cur_segment_sizes[i] - 1))
                frm_context_num.extend([0] * (cur_segment_sizes[i] - 1))
                if frm_context[i] == 0:
                    frm_salient_num.append(min(frm_salient[i], frm_len))
                    frm_context_num.append(0)
                elif frm_context[i] > 0:
                    frm_context_num.append(min(frm_context[i], frm_len // 2))
                    frm_salient_num.append(min(frm_salient[i], frm_len - frm_context_num[-1]))
                else:
                    frm_salient_num.append(frm_len - int(frm_len * d))
                    frm_context_num.append(int(frm_len * d))

        salient, contexts = [], []
        for f in range(cur_seg_len):
            top = torch.topk(cur_attn[f], frm_salient_num[f]).indices if frm_salient_num[f] > 0 else None
            if frm_context_num[f] > 0:
                all_idx = torch.arange(frm_len, device=device)
                remaining = all_idx[~torch.isin(all_idx, top)] if top is not None else all_idx
                x = cur_feat[f][remaining].unsqueeze(0).float()
                dist = torch.cdist(x, x) / (x.shape[-1] ** 0.5)
                # Density peaks (DPC-kNN): local density from the k nearest neighbours, distance to denser tokens.
                nearest, _ = torch.topk(dist, k=k, dim=-1, largest=False)
                density = (-(nearest ** 2).mean(dim=-1)).exp()
                density = density + torch.rand(density.shape, generator=gen, device=device, dtype=density.dtype) * 1e-6
                mask = (density[:, None, :] > density[:, :, None]).to(x.dtype)
                dist_max = dist.flatten(1).max(dim=-1)[0][:, None, None]
                delta, _ = (dist * mask + dist_max * (1 - mask)).min(dim=-1)
                _, sampled = torch.topk(delta * density, k=frm_context_num[f], dim=-1)
                contexts.append(remaining[sampled[0].sort().values] + f * frm_len)
            if top is not None:
                salient.append(top + f * frm_len)

        flat = cur_feat.reshape(cur_seg_len * frm_len, -1)
        unit = flat.float() / flat.float().norm(dim=-1, keepdim=True)
        salient = torch.cat(salient) if salient else torch.empty(0, dtype=torch.long, device=device)
        all_idx = torch.arange(cur_seg_len * frm_len, device=device)
        merges = []
        for ctx in contexts:
            merge_idx = all_idx[~torch.isin(all_idx, torch.cat([salient, ctx]))]
            similarity = unit[merge_idx] @ unit[ctx].T
            one_hot = torch.zeros(merge_idx.numel(), ctx.numel(), device=device, dtype=torch.float32)
            one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)
            weights = (1 / (one_hot.sum(dim=0).unsqueeze(-1) + 1)).clamp(min=beta)
            merges.append((ctx, merge_idx, one_hot, weights))
        plan.append((start, cur_seg_len, salient, merges))
        start += cur_seg_len

    keep = [[] for _ in range(num_frames)]
    for first, length, salient, merges in plan:
        idx = torch.cat([salient] + [m[0] for m in merges]).sort().values
        for i in idx.tolist():
            keep[first + i // frm_len].append(i % frm_len)

    def apply(tensor):
        """Kept tokens of an aligned tensor [G, N, ...], with anchors replaced by their merged values; per group."""
        out = [[] for _ in range(num_frames)]
        for first, length, salient, merges in plan:
            flat = tensor[first:first + length].reshape(length * frm_len, *tensor.shape[2:])
            idx, vals = [salient], [flat[salient]]
            for ctx, merge_idx, one_hot, weights in merges:
                counts = one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                agg = (one_hot.T @ flat[merge_idx].float()) / counts
                vals.append((weights * flat[ctx].float() + (1 - weights) * agg).to(flat.dtype))
                idx.append(ctx)
            idx, vals = torch.cat(idx), torch.cat(vals)
            order = torch.argsort(idx)
            idx, vals = idx[order], vals[order]
            for g in range(length):
                sel = (idx // frm_len) == g
                out[first + g].append(vals[sel])
        return [torch.cat(v) if v else tensor.new_zeros((0, *tensor.shape[2:])) for v in out]

    return [torch.as_tensor(sorted(k_), dtype=torch.long) for k_ in keep], apply
