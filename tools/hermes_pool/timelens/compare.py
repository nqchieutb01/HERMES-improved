"""Phase 37 table: TimeLens-Bench QVHighlights subset (shard 01, 513 queries), TimeLens metric (first predicted span,
rounded to 1 s; R1@{0.3,0.5,0.7}, mIoU). Paired 95% bootstrap (2,000 resamples) against the TimeLens-script baseline
at the same budget; * = excludes 0. Kept visual tokens: TimeLens script = its video tokens (timestamps not counted);
ours = kept tokens minus the timestamp / frame-boundary tokens of the surviving frame pairs (from the logs).
Usage: python3 compare.py
"""
import glob
import json
import random
import re
import sys

T = "/home/chieu.nguyen/hermes_pool/timelens"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/timelens_qvh/"
sys.path.insert(0, f"{T}/repo")
from timelens.utils import iou  # noqa: E402


def load(path):
    d = {}
    for line in open(path):
        d.update(json.loads(line))
    out = {}
    for key, pred in d.items():
        gt = eval(key.split(">>>")[2])
        ts = pred["timestamps"][0] if pred["timestamps"] else (-100, -100)
        out[key] = iou(gt, ts)
    return out


def kept_visual(run):
    """Mean kept visual tokens of one of our runs: kept - (all - visual) over the chunk logs."""
    vals = []
    for log in glob.glob(f"{R}{run}/inference-*.log"):
        text = open(log, errors="ignore").read()
        per_pair = [int(x) for x in re.findall(r": (\d+) visual tokens per temporal group", text)]
        for kept, total in re.findall(r"Offline pruning: keeping (\d+) of (\d+)", text):
            vals.append((int(kept), int(total)))
    if not vals:
        return None
    return vals


def metrics(ious):
    v = list(ious.values())
    return [100 * sum(x >= t for x in v) / len(v) for t in (0.3, 0.5, 0.7)] + [100 * sum(v) / len(v)]


def delta(a, b, n=2000):
    keys = sorted(a)
    d = [b[k] - a[k] for k in keys]
    rng = random.Random(0)
    bs = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    m, lo, hi = 100 * sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n)]
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"


TL = lambda tag: f"{T}/runs/{tag}/all.jsonl"  # noqa: E731
OURS = lambda tag: f"{R}{tag}/timelens_pred.jsonl"  # noqa: E731
ROWS = [  # (budget, label, path, token note, reference label)
    ("full", "Qwen3-VL-8B, TimeLens script (anchor)", TL("anchor"), "13,650", None),
    ("full", "Ours, unpruned (bridge)", OURS("bridge-unpruned"), "13,650", "anchor"),
    ("~4.2k", "TimeLens script 2 fps, lower resolution", TL("tl-2fps-4500"), "4,200", None),
    ("~4.2k", "TimeLens script 1 fps", TL("tl-1fps-5000"), "4,500", "tl2"),
    ("~4.2k", "Random, 300 frames", OURS("p300-random31"), "3,271", "tl2"),
    ("~4.2k", "SemVID, 300 frames", OURS("p300-semvid31"), "4,232", "tl2"),
    ("~4.2k", "Cover, 300 frames", OURS("p300-cover31"), "3,271", "tl2"),
    ("~4.2k", "Cover + PCD, 300 frames", OURS("p300-cover31-pcd"), "3,271", "tl2"),
    ("~4.2k", "Retrieve + random, 128 frames", OURS("rc128-random43"), "3,876", "tl2"),
    ("~4.2k", "Retrieve + Cover, 128 frames", OURS("rc128-cover43"), "3,876", "tl2"),
    ("~4.2k", "Retrieve + Cover + PCD, 128 frames", OURS("rc128-cover43-pcd"), "3,876", "tl2"),
    ("~1.1k", "TimeLens script 1 fps, lower resolution", TL("tl-1fps-1365"), "1,125", None),
    ("~1.1k", "Random, 300 frames (timestamps fill the budget)", OURS("p300-random8"), "~0", "tl1"),
    ("~1.1k", "Retrieve + random, 64 frames", OURS("rc64-random23"), "897", "tl1"),
    ("~1.1k", "Retrieve + Cover, 64 frames", OURS("rc64-cover23"), "897", "tl1"),
    ("~1.1k", "Retrieve + Cover + PCD, 64 frames", OURS("rc64-cover23-pcd"), "897", "tl1"),
]
REFS = {"anchor": TL("anchor"), "tl2": TL("tl-2fps-4500"), "tl1": TL("tl-1fps-1365")}

if __name__ == "__main__":
    refs = {k: load(v) for k, v in REFS.items()}
    print("| Budget | Method | Visual tokens | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | ΔmIoU vs budget baseline |")
    print("|---|---|---|---|---|---|---|---|")
    for budget, label, path, tokens, ref in ROWS:
        try:
            ious = load(path)
        except FileNotFoundError:
            print(f"| {budget} | {label} | {tokens} | pending | | | | |")
            continue
        m = metrics(ious)
        d = delta(refs[ref], ious) if ref else "–"
        print(f"| {budget} | {label} | {tokens} | " + " | ".join(f"{x:.1f}" for x in m) + f" | {d} |")
