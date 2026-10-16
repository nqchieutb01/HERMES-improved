"""A3: decompose grounding interval-length error into model bias and pruning-induced shrinkage (no GPU).

For each run: median predicted length, median gold length, median length ratio (pred / gold), share of
predictions shorter than half the gold length, and mIoU, overall and by question category. Runs with no
pruning (uniform sampling) show the model's own bias; the gap to pruned runs is what pruning adds.
Offline-pruning runs (phase 9 A1) are picked up automatically once scored.
Run from the repo root: python3 logs/phase9/extent_bias.py
"""
import glob
import json
import os
import statistics as st

B = "results/qwen3_vl_8b/sember_grounding"
KS = "-attention_weighted-native-time-keepsurv"
RUNS = [
    ("uniform 64 (no pruning)", "uniform-n64-time-count-location-v300-native-time"),
    ("uniform 32 (no pruning)", "uniform-n32-time-count-location-v300-native-time"),
    ("stream 0.2fps kv6000", f"time-count-location-fps0.2-kv6000-k0{KS}-v300"),
    ("stream 0.2fps kv4000", f"time-count-location-fps0.2-kv4000-k0{KS}-v300"),
    ("stream 0.2fps kv21400", f"time-count-location-fps0.2-kv21400-k0{KS}-v300"),
    ("stream grid32x8 kv6000", f"time-count-location-grid32x8-kv6000-k0{KS}-v300"),
]
for d in sorted(glob.glob(f"{B}/uniform-n64-time-count-location-offline-*-v300-native-time")) + \
        sorted(glob.glob(f"{B}/uniform-n64-time-count-location-scale*-v300-native-time")):
    tag = os.path.basename(d)
    RUNS.append((tag.replace("uniform-n64-time-count-location-", "uni64 ").replace("-v300-native-time", ""), tag))
CATS = [("time_duration", "time"), ("counting_objects_events", "count"), ("location_trace", "location")]


def load(tag):
    path = os.path.join(B, tag, "sember_grounding_scored.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else None


def stats(rows):
    ok = [r for r in rows if r.get("pred_start_time") not in (None, "")]
    if not ok:
        return None
    pl = [float(r["pred_end_time"]) - float(r["pred_start_time"]) for r in ok]
    gl = [float(r["answer_end_time"]) - float(r["answer_start_time"]) for r in ok]
    ratio = [max(p, 1.0) / max(g, 1.0) for p, g in zip(pl, gl)]
    miou = 100 * sum(float(r["temporal_iou"] or 0) for r in rows) / len(rows)
    return st.median(pl), st.median(gl), st.median(ratio), 100 * sum(x < 0.5 for x in ratio) / len(ratio), miou


def main():
    print("median pred len / median gold len / median ratio / % shorter than half gold / mIoU")
    header = f"{'run':34s} {'all':>34s}" + "".join(f"{c:>34s}" for _, c in CATS)
    print(header)
    for name, tag in RUNS:
        rows = load(tag)
        if rows is None:
            continue
        cells = []
        for key in [None] + [k for k, _ in CATS]:
            s = stats([r for r in rows if key is None or r.get("question_category") == key])
            cells.append("--" if s is None else f"{s[0]:5.1f} / {s[1]:4.0f} / {s[2]:4.2f} / {s[3]:3.0f}% / {s[4]:4.1f}")
        print(f"{name:34s} " + "".join(f"{c:>34s}" for c in cells))


if __name__ == "__main__":
    main()
