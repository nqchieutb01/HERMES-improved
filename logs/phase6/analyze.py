"""Phase 6 analysis: early-dense sampling schedules vs fixed 0.2 fps and uniform baselines (Qwen3, 300 videos).

For each finished run prints MCQ Acc, grounding mIoU and R@0.5 overall and per question-time
bucket, plus the paired-bootstrap 95% CI of the difference from the fps0.2 KV6000 k0 baseline.
Run from the repo root: python3 logs/phase6/analyze.py
"""
import glob
import json
import os
import random

B = "results/qwen3_vl_8b"
BASE_TAG = "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300"
FIXED = [
    ("uniform 32", "uniform-n32-time-count-location-v300-native-time"),
    ("uniform 64", "uniform-n64-time-count-location-v300-native-time"),
    ("fps0.2 kv6000 (base)", BASE_TAG),
    ("fps0.2 kv10700", "time-count-location-fps0.2-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
]
BINS = [(0, 80), (80, 160), (160, 320), (320, 1e9)]


def truthy(x):
    return x in (True, 1, "True", "true", "1", "1.0")


def load(task, tag):
    path = f"{B}/{task}/{tag}/{task}_scored.jsonl"
    if not os.path.exists(path):
        return None
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if task == "sember_mcq":
            v = [float(truthy(r["is_correct"]))]
        else:
            v = [float(r["temporal_iou"] or 0), float(truthy(r["recall_at_1_iou_0.5"]))]
        rows[r["question_id"]] = (float(r["question_time"]), v)
    return rows


def methods():
    out = list(FIXED)
    for d in sorted(glob.glob(f"{B}/sember_*/time-count-location-[sg]*-v300")):
        tag = os.path.basename(d)
        if (os.path.basename(os.path.dirname(d)), tag) and tag not in [t for _, t in out]:
            out.append((tag.replace("time-count-location-", "").replace("-attention_weighted-native-time-keepsurv-v300", ""), tag))
    for d in sorted(glob.glob(f"{B}/sember_*/time-count-location-fps1.0-kv*-k0-attention_weighted-native-time-keepsurv-v300")):
        tag = os.path.basename(d)
        if tag not in [t for _, t in out]:
            out.append((tag.replace("time-count-location-", "").replace("-attention_weighted-native-time-keepsurv-v300", ""), tag))
    return out


def boot_ci(a, b, n=2000, seed=0):
    """95% CI of mean(a - b) over paired questions."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def main():
    ms = methods()
    data = {(task, tag): load(task, tag) for _, tag in ms for task in ("sember_mcq", "sember_grounding")}
    cols = [("sember_mcq", 0, "Acc"), ("sember_grounding", 0, "mIoU"), ("sember_grounding", 1, "R@.5")]
    buckets = [("all", 0, 1e9)] + [(f"{lo}-{hi:g}" if hi < 1e8 else f">={lo}", lo, hi) for lo, hi in BINS]
    hdr = f"{'method':34s}" + "".join(f"{b:>20s}" for b, _, _ in buckets)
    print("Acc / mIoU / R@0.5 per question-time bucket (%)  -- = run missing or unfinished")
    print(hdr)
    for name, tag in ms:
        cells = []
        for _, lo, hi in buckets:
            parts = []
            for task, i, _ in cols:
                rows = data[task, tag]
                if rows is None:
                    parts.append("  --")
                    continue
                vals = [v[i] for t, v in rows.values() if lo <= t < hi]
                parts.append(f"{100 * sum(vals) / len(vals):4.1f}" if vals else "  --")
            cells.append("/".join(parts))
        print(f"{name:34s}" + "".join(f"{c:>20s}" for c in cells))

    print(f"\nDifference from fps0.2 kv6000 base, overall, with paired-bootstrap 95% CI (points)")
    for name, tag in ms:
        if tag == BASE_TAG:
            continue
        cells = []
        for task, i, label in cols:
            rows, base = data[task, tag], data[task, BASE_TAG]
            if rows is None:
                cells.append(f"{label} --")
                continue
            q = sorted(set(rows) & set(base))
            a, b = [rows[x][1][i] for x in q], [base[x][1][i] for x in q]
            lo, hi = boot_ci(a, b)
            cells.append(f"{label} {100 * (sum(a) - sum(b)) / len(q):+5.1f} [{lo:+.1f},{hi:+.1f}]")
        print(f"{name:34s}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
