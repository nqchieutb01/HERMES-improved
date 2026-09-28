"""M3: retrieval-based boundary refinement for grounding under token pruning (no GPU).

A per-frame embedding index (logs/phase10/dump_frame_embeddings.py) lives outside the KV cache, so
pruning never touches it. After the model answers with [s, e]:
1. use only index frames up to the question time (streaming-legal); subtract their mean embedding
2. evidence profile r = mean embedding of index frames inside [s, e] (nearest frame if none)
3. grow each boundary outward while neighbouring frames stay similar to r
   (cosine >= tau * mean cosine of the evidence frames), tolerating gaps of up to `gap` seconds

tau and gap are fitted on half of the videos and scored on the other half (2-fold by video, swapped),
the same protocol as logs/phase8/calibration_method.py. Compared with raw predictions, the fixed x4
widening, and a fitted length scale.
Run from the repo root with numpy available, e.g.
  /nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3 logs/phase10/boundary_refine.py [--index_fps 0.2]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "logs/phase8")
import calibration_method as cm  # noqa: E402

EMB = "results/qwen3_vl_8b/frame_embeddings/fps1-scale0.5"
KS = "-attention_weighted-native-time-keepsurv"
RUNS = [
    ("uniform 32 (no pruning)", "uniform-n32-time-count-location-v300-native-time"),
    ("uniform 64 (no pruning)", "uniform-n64-time-count-location-v300-native-time"),
    ("uni64 offline stratified 25%", "uniform-n64-time-count-location-offline-stratified-keep0.25-v300-native-time"),
    ("uni64 offline hermes 10%", "uniform-n64-time-count-location-offline-hermes-keep0.1-v300-native-time"),
    ("stream 0.2fps kv6000", f"time-count-location-fps0.2-kv6000-k0{KS}-v300"),
    ("stream 0.2fps kv4000", f"time-count-location-fps0.2-kv4000-k0{KS}-v300"),
    ("stream grid16x16 kv6000", f"time-count-location-grid16x16-kv6000-k0{KS}-v300"),
]
TAUS = [round(0.1 * i, 1) for i in range(11)]            # 0.0 .. 1.0
GAPS = [0, 5, 10, 20, 40, 60]                          # seconds


def load_index(video_id, index_fps):
    d = np.load(os.path.join(EMB, f"{video_id}.npz"))
    times, emb = d["times"].astype(np.float64), d["emb"].astype(np.float32)
    step = max(1, int(round(1.0 / index_fps)))             # subsample the 1 fps index
    return times[::step], emb[::step]


def prepare(rows_raw, index_fps):
    """Attach per-question similarity profiles (computed once; tau/gap search is then cheap)."""
    cache, out = {}, []
    for r, raw in rows_raw:
        vid = raw["video_id"]
        if vid not in cache:
            cache[vid] = load_index(vid, index_fps)
        times, emb = cache[vid]
        past = times <= r["qt"] + 1e-6
        t, e = times[past], emb[past]
        if not r["ok"] or len(t) < 2:
            out.append(dict(r, sim=None))
            continue
        e = e - e.mean(axis=0, keepdims=True)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        inside = (t >= r["ps"]) & (t <= r["pe"])
        if not inside.any():
            inside[np.argmin(np.abs(t - (r["ps"] + r["pe"]) / 2))] = True
        ref = e[inside].mean(axis=0)
        ref /= np.linalg.norm(ref) + 1e-8
        sim = e @ ref
        idx = np.nonzero(inside)[0]
        out.append(dict(r, sim=sim, t=t, lo=int(idx[0]), hi=int(idx[-1]), ev=float(sim[inside].mean())))
    return out


def refine(q, tau, gap):
    if q["sim"] is None:
        return q["ps"], q["pe"]
    sim, t, thr = q["sim"], q["t"], tau * q["ev"]
    step = (t[1] - t[0]) if len(t) > 1 else 1.0
    max_miss = int(round(gap / step))
    lo, miss, i = q["lo"], 0, q["lo"] - 1
    while i >= 0 and miss <= max_miss:
        if sim[i] >= thr:
            lo, miss = i, 0
        else:
            miss += 1
        i -= 1
    hi, miss, i = q["hi"], 0, q["hi"] + 1
    while i < len(t) and miss <= max_miss:
        if sim[i] >= thr:
            hi, miss = i, 0
        else:
            miss += 1
        i += 1
    return min(q["ps"], t[lo]), max(q["pe"], min(t[hi] + step, q["qt"]))


def score(qs, fn):
    ious = []
    for q in qs:
        if not q["ok"]:
            ious.append(0.0)
            continue
        s, e = fn(q)
        ious.append(cm.iou(s, e, q["gs"], q["ge"]) if e > s else 0.0)
    return 100 * np.mean(ious), 100 * np.mean([x >= 0.5 for x in ious])


def cv(qs, candidates, make_fn):
    tot, n, fits = np.zeros(2), 0, []
    for fold in (0, 1):
        train = [q for q in qs if q["fold"] != fold]
        test = [q for q in qs if q["fold"] == fold]
        best = max(candidates, key=lambda c: score(train, make_fn(*c))[0])
        fits.append(best)
        tot += np.array(score(test, make_fn(*best))) * len(test)
        n += len(test)
    return tot / n, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_fps", type=float, default=1.0)
    args = ap.parse_args()
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(EMB, "*.npz"))}
    print(f"index: {len(have)} videos at {args.index_fps} fps")
    print(f"{'method':32s} {'n':>4s} {'raw':>13s} {'x4 fixed':>13s} {'scale (cv)':>13s} {'M3 (cv)':>13s} {'M3 then x2':>13s}  M3 fits")
    for name, tag in RUNS:
        path = os.path.join(cm.B, tag, "sember_grounding_scored.jsonl")
        if not os.path.exists(path):
            continue
        raws = [cm.json.loads(line) for line in open(path)]
        rows = cm.load(tag)
        pairs = [(r, raw) for r, raw in zip(rows, raws) if raw["video_id"] in have]
        if not pairs:
            continue
        qs = prepare(pairs, args.index_fps)
        raw = score(qs, lambda q: (q["ps"], q["pe"]))
        x4 = score(qs, lambda q: cm.apply(q, "scale", (4, 0)))
        sc, _ = cv(qs, [(a / 4,) for a in range(2, 21)], lambda a: (lambda q: cm.apply(q, "scale", (a, 0))))
        m3, fits = cv(qs, [(t, g) for t in TAUS for g in GAPS], lambda t, g: (lambda q: refine(q, t, g)))

        def m3x2(t, g):
            def fn(q):
                s, e = refine(q, t, g)
                c, half = (s + e) / 2, (e - s)
                return max(0.0, c - half), min(q["qt"], c + half)
            return fn
        m3s, _ = cv(qs, [(t, g) for t in TAUS for g in GAPS], m3x2)
        cells = [raw, x4, sc, m3, m3s]
        print(f"{name:32s} {len(qs):4d} " + " ".join(f"{c[0]:5.1f} / {c[1]:5.1f}" for c in cells) + f"  {fits}")


if __name__ == "__main__":
    main()
