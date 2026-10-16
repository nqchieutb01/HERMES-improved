"""A2: which frames' tokens survive pruning, and does it predict grounding success? (no GPU)

Joins per-question retention snapshots (retention-<chunk>.jsonl, written with run.retention_snapshot=true)
with scored grounding predictions. For each question:
- evidence frames: streamed frames whose time lies in the annotated interval [start, end]
  (the nearest frame when the interval falls between two samples)
- keep rate of a frame: kept visual tokens / encoded tokens, averaged over layers
- edge frames: first and last evidence frame; interior: the rest; background: frames outside the interval
- timestamp kept: whether the frame's temporal group still has its timestamp tokens

Reports keep rates for background / evidence / edge / interior frames, and grounding (mIoU, R@0.5,
predicted length) split by how much of the evidence survived.
Run from the repo root: python3 logs/phase9/retention_analysis.py
"""
import glob
import json
import os
import statistics as st

B = "results/qwen3_vl_8b/sember_grounding"
RUNS = [
    ("0.2fps kv6000", "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-retention-v300"),
    ("0.2fps kv4000", "time-count-location-fps0.2-kv4000-k0-attention_weighted-native-time-keepsurv-retention-v300"),
    ("grid32x8 kv6000", "time-count-location-grid32x8-kv6000-k0-attention_weighted-native-time-keepsurv-retention-v300"),
]


def load(run_dir):
    snaps = {}
    for path in glob.glob(os.path.join(run_dir, "retention-*.jsonl")):
        for line in open(path):
            s = json.loads(line)
            snaps[s["question_id"]] = s
    scored = {}
    path = os.path.join(run_dir, "sember_grounding_scored.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            scored[r["question_id"]] = r
    return snaps, scored


def frame_rates(snap):
    """frame index -> (keep rate, timestamp kept) for every streamed frame."""
    enc, kept, tkept = snap["encoded"], snap["kept"], snap["time_kept"]
    out = {}
    for f_str, n in enc.items():
        f = int(f_str)
        group = f - (f % 2)  # Qwen temporal groups start on even stream frames within aligned chunks
        out[f] = (float(kept.get(f_str, 0.0)) / n if n else 0.0,
                  float(tkept.get(str(group), tkept.get(str(f), 0.0))) > 0)
    return out


def question_stats(snap, r):
    times = snap["frame_times"]
    gs, ge = float(r["answer_start_time"]), float(r["answer_end_time"])
    inside = [i for i, t in enumerate(times) if gs <= t <= ge]
    if not inside and times:
        mid = (gs + ge) / 2
        inside = [min(range(len(times)), key=lambda i: abs(times[i] - mid))]
    rates = frame_rates(snap)
    ev = [rates[i][0] for i in inside if i in rates]
    edges = {inside[0], inside[-1]} if inside else set()
    edge = [rates[i][0] for i in edges if i in rates]
    interior = [rates[i][0] for i in inside if i in rates and i not in edges]
    bg = [v[0] for i, v in rates.items() if i not in set(inside)]
    ts = [rates[i][1] for i in inside if i in rates]
    ok = r.get("pred_start_time") not in (None, "")
    return {
        "ev": st.mean(ev) if ev else None, "edge": st.mean(edge) if edge else None,
        "interior": st.mean(interior) if interior else None, "bg": st.mean(bg) if bg else None,
        "ts": (sum(ts) / len(ts)) if ts else None, "survived": any(x > 0 for x in ev),
        "iou": float(r["temporal_iou"] or 0), "r05": r["recall_at_1_iou_0.5"] in (True, "True", "true", 1),
        "plen": float(r["pred_end_time"]) - float(r["pred_start_time"]) if ok else None,
        "glen": ge - gs,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    for name, tag in RUNS:
        run_dir = os.path.join(B, tag)
        snaps, scored = load(run_dir)
        qs = [question_stats(snaps[q], scored[q]) for q in snaps if q in scored]
        if not qs:
            print(f"{name}: no data yet")
            continue
        print(f"\n== {name}  ({len(qs)} questions)")
        print(f"keep rate  background {100 * mean([q['bg'] for q in qs]):5.1f}%   evidence {100 * mean([q['ev'] for q in qs]):5.1f}%"
              f"   edge {100 * mean([q['edge'] for q in qs]):5.1f}%   interior {100 * mean([q['interior'] for q in qs]):5.1f}%"
              f"   evidence timestamps kept {100 * mean([q['ts'] for q in qs]):5.1f}%")
        ev_sorted = sorted(q["ev"] for q in qs if q["ev"] is not None)
        cut = [ev_sorted[len(ev_sorted) // 3], ev_sorted[2 * len(ev_sorted) // 3]] if ev_sorted else [0, 0]
        groups = [("no evidence tokens left", lambda q: not q["survived"]),
                  ("evidence kept: low third", lambda q: q["survived"] and q["ev"] is not None and q["ev"] <= cut[0]),
                  ("evidence kept: mid third", lambda q: q["ev"] is not None and cut[0] < q["ev"] <= cut[1]),
                  ("evidence kept: top third", lambda q: q["ev"] is not None and q["ev"] > cut[1])]
        print(f"{'group':28s} {'n':>4s} {'mIoU':>6s} {'R@0.5':>6s} {'pred/gold len':>14s}")
        for label, pred in groups:
            g = [q for q in qs if pred(q)]
            if not g:
                continue
            ratio = [max(q["plen"], 1) / max(q["glen"], 1) for q in g if q["plen"] is not None]
            print(f"{label:28s} {len(g):4d} {100 * mean([q['iou'] for q in g]):6.1f} {100 * mean([q['r05'] for q in g]):6.1f}"
                  f" {st.median(ratio) if ratio else float('nan'):14.2f}")


if __name__ == "__main__":
    main()
