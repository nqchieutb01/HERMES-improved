"""Paired bootstrap between scored open-ended runs (score.py --open writes nextgqa_open_eval.json): A - B for
Open-Acc@QA, Open-Acc&GQA (judged correct and IoP >= 0.5), mIoP, mIoU; 2,000 resamples, seed 0.
Usage: python paired_open.py <run B (reference)> <run A> [<run A2> ...]
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


def per_q(run):
    e = json.load(open(f"{run}/nextgqa_open_eval.json"))
    out = {}
    for k, p in e["pred_ground"].items():
        iou = max(0, max(get_tIoU(g, p)[0] for g in SPANS[k]))
        iop = max(0, max(get_tIoU(g, p)[1] for g in SPANS[k]))
        ok = e["pred_qa"][k]["prediction"] == "yes"
        out[k] = (float(ok), float(ok and iop >= 0.5), iop, iou)
    return out


ref = per_q(sys.argv[1])
names = ["Acc", "Acc&GQA", "mIoP", "mIoU"]
print(f"{'run':24s} " + " ".join(f"{n:>8s}" for n in names) + "   (paired vs " + sys.argv[1].rstrip('/').split('/')[-1] + ")")
for run in sys.argv[1:]:
    m = per_q(run)
    ks = sorted(m)
    vals = " ".join(f"{100 * sum(m[k][i] for k in ks) / len(ks):8.1f}" for i in range(4))
    extra = ""
    if run != sys.argv[1]:
        rng = random.Random(0)
        for i, n in ((0, "Acc"), (1, "GQA")):
            d = [m[k][i] - ref[k][i] for k in ks]
            bs = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
            star = "*" if bs[50] > 0 or bs[1949] < 0 else ""
            extra += f" | d{n} {100 * sum(d) / len(d):+.1f} [{bs[50]:+.1f}, {bs[1949]:+.1f}]{star}"
    print(f"{run.rstrip('/').split('/')[-1]:24s} {vals}{extra}")
