"""Phase 13: timeline-reasoning prompt vs the official prompt, paired per question (no GPU).

For every phase 13 run, its baseline is the same run directory without "-timeline". Reports answer accuracy
(official S-EMBER judge prompt), mIoU, R@0.5 and GQ@0.5 for both, and the timeline-minus-official
difference with a 95% paired bootstrap CI (10k resamples over questions). "mIoU m:ss" also reads intervals
written as minutes:seconds ("Time: [1:29, 2:58]"), which the official parser scores as no interval.
Run from the repo root: python3 logs/phase13/compare.py
"""
import json
import os
import random
import re

RUNS = "logs/phase13/judge_runs.txt"


def iou(a, b, c, d):
    inter, union = max(0.0, min(b, d) - max(a, c)), max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0


def tolerant_iou(r):
    """IoU that also parses m:ss intervals; equals the official IoU when the official parser succeeded."""
    if r["interval_parseable"]:
        return float(r["temporal_iou"] or 0)
    m = re.search(r"Time:\s*\[\s*(\d+):(\d\d(?:\.\d+)?)\s*,\s*(\d+):(\d\d(?:\.\d+)?)", r.get("pred_raw") or "")
    if not m:
        return 0.0
    s, e = int(m[1]) * 60 + float(m[2]), int(m[3]) * 60 + float(m[4])
    return iou(s, e, float(r["answer_start_time"]), float(r["answer_end_time"]))


def per_question(run):
    d = f"results/{run}"
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{d}/sember_grounding_scored.jsonl")}
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{d}/answer_judgments.jsonl")}
    out = {}
    for q, r in rows.items():
        i = float(r["temporal_iou"] or 0)
        out[q] = {"acc": judge[q], "miou": i, "r05": i >= 0.5, "gq05": judge[q] and i >= 0.5,
                  "miou_mmss": tolerant_iou(r)}
    return out


def boot(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    keys = ["acc", "miou", "r05", "gq05", "miou_mmss"]
    names = {"acc": "Acc.", "miou": "mIoU", "r05": "R@0.5", "gq05": "GQ@0.5", "miou_mmss": "mIoU m:ss"}
    runs = [l.strip() for l in open(RUNS) if "-timeline" in l]
    for run in runs:
        base = run.replace("-timeline", "")
        if not all(os.path.exists(f"results/{r}/answer_judgments.jsonl") for r in (run, base)):
            print(f"{run}: not judged yet")
            continue
        t, b = per_question(run), per_question(base)
        qs = sorted(set(t) & set(b))
        print(f"\n{run.split('/')[0]}  {os.path.basename(run).replace('-timeline', '')}  (n={len(qs)})")
        for k in keys:
            tv = [100 * float(t[q][k]) for q in qs]
            bv = [100 * float(b[q][k]) for q in qs]
            diffs = [x - y for x, y in zip(tv, bv)]
            lo, hi = boot(diffs)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {names[k]:10s} official {sum(bv) / len(qs):5.1f}  timeline {sum(tv) / len(qs):5.1f}  "
                  f"diff {sum(diffs) / len(qs):+5.1f} [{lo:+5.1f}, {hi:+5.1f}] {sig}")


if __name__ == "__main__":
    main()
