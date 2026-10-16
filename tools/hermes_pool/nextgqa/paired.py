"""Paired bootstrap (2,000 resamples, seed 0) between two scored NExT-GQA runs (score.py writes nextgqa_eval.json).
Usage: python paired.py <run A> <run B>   -> A - B for Acc&GQA, Acc@QA, mIoP, mIoU with 95% CIs.
"""
import json
import random
import sys

sys.path.insert(0, "/home/chieu.nguyen/hermes_pool/nextgqa/repo/code/TempGQA")
from eval_ground import get_tIoU  # noqa: E402

SPANS = {}
for line in open("/nfs-stor/chieu.nguyen/nextgqa/nextgqa_test_subset.jsonl"):
    m = json.loads(line)["nextgqa"]
    SPANS[f"{m['vid']}_{m['qid']}"] = m["spans"]


def per_question(run):
    e = json.load(open(f"{run}/nextgqa_eval.json"))
    out = {}
    for k, p in e["pred_ground"].items():
        iou = max(0, max(get_tIoU(g, p)[0] for g in SPANS[k]))  # official: per-question max starts at 0
        iop = max(0, max(get_tIoU(g, p)[1] for g in SPANS[k]))
        ok = e["pred_qa"][k]["answer"] == e["pred_qa"][k]["prediction"]
        out[k] = (float(ok and iop >= 0.5), float(ok), iop, iou)
    return out


a, b = per_question(sys.argv[1]), per_question(sys.argv[2])
keys = sorted(a)
rng = random.Random(0)
for i, name in enumerate(["Acc&GQA", "Acc@QA", "mIoP", "mIoU"]):
    d = [a[k][i] - b[k][i] for k in keys]
    boot = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    print(f"{sys.argv[1].rstrip('/')} - {sys.argv[2].rstrip('/')} {name}: {100 * sum(d) / len(d):+.1f} "
          f"[{boot[50]:+.1f}, {boot[1949]:+.1f}]")
