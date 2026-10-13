"""FlashVID (Fan et al., ICLR 2026): attention-and-diversity token selection + tree-based spatiotemporal merging.

Training-free visual token compression applied to the vision encoder's output before the LLM. This is a line-by-line
port of the authors' vision-side compression (github.com/Fanziyang-v/FlashVID, flashvid/utils.py ``flashvid_compression``
and flashvid/token_selection.py), with the hyperparameters of their Qwen3-VL demo (playground/qwen3_vl.py): dynamic
segmentation threshold 0.9 with at least 4 segments, alpha 0.7 (share of ADTS tokens), ADTS v1, temporal threshold 0.8.
A "frame" is one temporal group of Qwen3-VL, as in the original.

Deviation: FlashVID's second stage (FastV pruning inside the LLM, ranked by the attention of the prompt's last token)
is left out, because HERMES builds the memory before the question is known. The vision side therefore keeps the whole
budget (expansion 1.0 instead of 1.25), so the memory holds ``retention`` of the visual tokens in every layer, as for
the other selectors. DeepStack features are taken at the kept indices without merging, as in the original.

Inputs per video: ``features`` [G, N, D], ``cls_attention`` [G, N] (last vision block: attention received per merged
token, averaged over heads and queries), ``retention`` r.
Output: for each group, the kept token indices (sorted), a function returning the kept (merged) features, and a
function selecting the kept indices from any aligned tensor [G, N, ...] (DeepStack).
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

DEFAULTS = dict(segment_threshold=0.9, min_segment_num=4, complementary_segment=True, alpha=0.7,
                temporal_threshold=0.8, expansion=1.0)


def _pairwise_cosine_distances(x):
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    return 1.0 - torch.bmm(x, x.transpose(-1, -2))


def _adts(features, cls_attention, num_retained_tokens):
    """Attention-and-diversity-based token selection (ADTS v1, ``attn_div_based_token_selection``)."""
    original = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6
    bsz, n, d = features.shape
    dist = _pairwise_cosine_distances(features) * cls_attention.unsqueeze(1)
    keep = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist, k=2, dim=1, largest=False).values[:, 1, :]
    keep[:, 0] = torch.argmax(min_dist, dim=-1)
    for i in range(1, num_retained_tokens):
        sub = torch.gather(dist, dim=1, index=keep[:, :i].unsqueeze(-1).expand(-1, -1, n))
        min_dist = torch.min(sub, dim=1).values
        min_dist.scatter_(1, keep[:, :i], -1)
        keep[:, i] = torch.argmax(min_dist, dim=-1)
    keep = keep.sort().values
    return torch.gather(original, dim=1, index=keep.unsqueeze(-1).expand(-1, -1, d)), keep


def _segment(frame_features, segment_threshold, min_segment_num, complementary_segment):
    num_frames = frame_features.shape[0]
    normed = frame_features / frame_features.norm(p=2, dim=-1, keepdim=True)
    sims = torch.sum(normed[:-1] * normed[1:], dim=-1)
    cut = torch.where(sims < segment_threshold)[0]
    if cut.numel() + 1 < min_segment_num and complementary_segment:
        sims[sims < segment_threshold] = 1.0
        extra = torch.topk(sims, k=min(min_segment_num - cut.numel() - 1, sims.shape[0]), largest=False).indices
        cut = torch.cat([cut, extra]).sort().values
    padded = F.pad(cut, (1, 1), value=0)
    padded[0] = -1
    padded[-1] = num_frames - 1
    return torch.diff(padded, n=1, dim=0)


@torch.no_grad()
def _dpc_knn(features, num_clusters, k, valid_token_mask):
    invalid = ~valid_token_mask
    bsz, seq_len, feat_dim = features.shape
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)
    dists = torch.masked_fill(dists, invalid.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)
    nearest = torch.topk(dists, k=max(k), dim=-1, largest=False).values
    density = torch.empty((bsz, seq_len), device=features.device, dtype=features.dtype)
    for i in range(bsz):
        density[i] = torch.mean(-(nearest[i, :, :k[i]] ** 2), dim=-1).exp()
    density = torch.masked_fill(density, invalid, 0.0)
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    dist, _ = torch.min(torch.where(mask, dists, max_dist), dim=-1)
    score = dist * density
    cluster_idx, center_idx = [], []
    for i in range(bsz):
        k_i = num_clusters[i]
        if k_i == 0:
            center_idx.append(torch.tensor([], dtype=torch.long, device=features.device))
            cluster_idx.append(torch.zeros(seq_len, dtype=torch.long, device=features.device))
            continue
        cc = torch.topk(score[i], k=k_i, dim=-1).indices
        c = torch.argmin(torch.gather(dists[i], dim=-1, index=cc.unsqueeze(0).expand(seq_len, -1)), dim=-1)
        c.scatter_(dim=-1, index=cc, src=torch.arange(k_i).to(c))
        center_idx.append(cc)
        cluster_idx.append(c)
    return cluster_idx, center_idx


def _spatiotemporal_compression(video_features, temporal_threshold, token_mask, lower_bound):
    """Tree-based temporal merging: tokens similar to a token of the previous frame are folded into it (in place)."""
    num_frames, n, feat_dim = video_features.shape
    normed = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cos = torch.bmm(normed[1:], normed[:-1].transpose(1, 2))
    cos[~token_mask[1:].unsqueeze(-1).expand(-1, -1, n)] = -1.0
    cos[~token_mask[:-1].unsqueeze(1).expand(-1, n, -1)] = -1.0
    max_sims, max_idx = torch.max(cos, dim=-1)
    max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    max_idx = F.pad(max_idx, (0, 0, 1, 0), value=-1)
    counts = torch.ones(num_frames, n).to(video_features)
    mask = max_sims > temporal_threshold
    retaining = ~mask
    if retaining.int().sum() < lower_bound:
        soft = max_sims.view(-1).topk(k=(num_frames * n) - lower_bound).values[-1]
        soft = max(soft, -1.0 + 1e-6)
        mask = max_sims > soft
        retaining = ~mask
    for f in range(num_frames - 1, -1, -1):
        feats, cnt, idx = video_features[f], counts[f], max_idx[f]
        merge_here = feats[~mask[f]]
        if merge_here.numel() > 0:
            video_features[f][~mask[f]] = merge_here / cnt[~mask[f]].unsqueeze(-1).to(merge_here.dtype)
            counts[f][~mask[f]] = 1
        others = feats[mask[f]]
        if others.numel() > 0:
            anchors = idx[mask[f]]
            agg = torch.zeros((n, feat_dim), dtype=video_features.dtype, device=video_features.device)
            agg.scatter_add_(0, anchors.unsqueeze(-1).expand(-1, feat_dim), others)
            video_features[f - 1] += agg
            counts[f - 1] += torch.bincount(anchors, minlength=n).to(video_features.dtype)
            counts[f][mask[f]] = 0
    tokens, indices = [], []
    for f in range(num_frames):
        m = retaining[f] & token_mask[f]
        tokens.append(video_features[f][m])
        indices.append(torch.where(m)[0])
    return tokens, indices


def _segment_compression(seg_feat, seg_idx, seg_attn, n_adts, n_sttm, temporal_threshold):
    num_frames, n, feat_dim = seg_feat.shape
    sel_feat, sel_idx = _adts(seg_feat, seg_attn, n_adts)
    sel_global = seg_idx.gather(1, index=sel_idx).view(-1)
    mask = torch.ones(num_frames, n, dtype=torch.bool, device=seg_feat.device)
    mask.scatter_(1, sel_idx, False)
    num_other = n_sttm * num_frames
    if num_other > 0 and temporal_threshold < 1.0:
        if num_frames > 1:
            tm_tokens, tm_idx = _spatiotemporal_compression(seg_feat, temporal_threshold, mask,
                                                            (n_adts + n_sttm) * num_frames)
            tm_global = [seg_idx.view(num_frames, -1)[i][t] for i, t in enumerate(tm_idx)]
        else:
            tm_tokens, tm_global = [seg_feat[0]], [seg_idx[0]]
    else:
        tm_tokens, tm_global = [], []
    all_tokens, all_global = [sel_feat.view(-1, feat_dim)], [sel_global]
    if num_other > 0:
        current = sum(len(t) for t in tm_tokens)
        ratio = num_other / current
        if ratio < 1.0:
            max_n = max(len(t) for t in tm_tokens)
            batched = torch.zeros((len(tm_tokens), max_n, feat_dim), dtype=seg_feat.dtype, device=seg_feat.device)
            valid = torch.zeros((len(tm_tokens), max_n), dtype=torch.bool, device=seg_feat.device)
            n_clusters, ks = [], []
            for i, t in enumerate(tm_tokens):
                batched[i, :len(t)] = t
                valid[i, :len(t)] = True
                n_clusters.append(math.ceil(len(t) * ratio))
                ks.append(min(n_clusters[-1], 7))
            cluster_idx, center_idx = _dpc_knn(batched, n_clusters, ks, valid)
            for i, (t, g) in enumerate(zip(tm_tokens, tm_global)):
                nc = n_clusters[i]
                if nc > 0:
                    ci = cluster_idx[i][:len(t)]
                    agg = torch.zeros((nc, feat_dim), dtype=seg_feat.dtype, device=seg_feat.device)
                    agg.scatter_add_(0, ci.unsqueeze(-1).expand(-1, feat_dim), t)
                    agg = agg / torch.bincount(ci, minlength=nc).unsqueeze(-1).to(seg_feat.dtype)
                    all_tokens.append(agg)
                    all_global.append(g[center_idx[i]])
                else:
                    all_tokens.append(t)
                    all_global.append(g)
        else:
            all_tokens.extend(tm_tokens)
            all_global.extend(tm_global)
    return torch.cat(all_tokens, dim=0), torch.cat(all_global, dim=0)


@torch.no_grad()
def flashvid(features, cls_attention, retention, segment_threshold=0.9, min_segment_num=4,
             complementary_segment=True, alpha=0.7, temporal_threshold=0.8, expansion=1.0):
    num_frames, n, _ = features.shape
    work = features.clone()  # the merging modifies the features in place
    lengths = _segment(work.mean(1), segment_threshold, min_segment_num, complementary_segment)
    global_idx = torch.arange(num_frames * n, dtype=torch.long, device=features.device).view(num_frames, n)
    budget = math.ceil(n * retention * expansion)
    n_adts = math.ceil(budget * alpha)
    n_sttm = budget - n_adts
    tokens, indices, offset = [], [], 0
    for seg_len in lengths.tolist():
        t, g = _segment_compression(work[offset:offset + seg_len], global_idx[offset:offset + seg_len],
                                    cls_attention[offset:offset + seg_len], n_adts, n_sttm, temporal_threshold)
        tokens.append(t)
        indices.append(g)
        offset += seg_len
    tokens, indices = torch.cat(tokens), torch.cat(indices)
    order = indices.argsort()
    tokens, indices = tokens[order], indices[order]
    groups = indices // n
    keep = [indices[groups == g] % n for g in range(num_frames)]
    kept_feats = [tokens[groups == g] for g in range(num_frames)]

    def apply_index(tensor):
        return [tensor[g][keep[g]] for g in range(num_frames)]

    return [k.cpu() for k in keep], kept_feats, apply_index
