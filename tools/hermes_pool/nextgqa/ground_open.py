"""Grounding-only metrics for open-ended NExT-GQA runs (no judge needed): official tIoU/tIoP (eval_ground's get_tIoU,
per-question max over gold spans starting at 0), mIoP, IoP@0.5, mIoU, IoU@0.5, the share of intervals starting at 0 s,
and paired bootstrap differences against a reference run (same questions; 2,000 resamples, seed 0).
Usage: python ground_open.py <ref run dir> <run dir> [...]   (protocol from the folder name: oe-* or tl-*)
"""
import csv
import json
import random
import sys

sys.path.insert(0, "/home/chieu.nguyen/hermes_pool/nextgqa/repo/code/TempGQA")
sys.path.insert(0, "/home/chieu.nguyen/hermes_pool/timelens/repo")
sys.path.insert(0, "/home/chieu.nguyen/HERMES/eval/sember")
from eval_ground import get_tIoU  # noqa: E402
from timelens.utils import extract_time  # noqa: E402
from eval_grounding import parse_grounding_response  # noqa: E402

S = [json.loads(l) for l in open("/nfs-stor/chieu.nguyen/nextgqa/nextgqa_test_subset.jsonl") if "_ans\"" in l]


def per_q(run_dir):
    proto = run_dir.rstrip("/").split("/")[-1].split("-")[0]
    P = {r["question_id"]: r["pred_raw"] for r in csv.DictReader(open(f"{run_dir}/results.csv"))}
    out = {}
    for s in S:
        base = s["question_id"][:-4]
        if proto == "oe":
            t = extract_time(P.get(base + "_grd", ""))
            sp = list(t[0]) if t else None
        else:
            _, a, b = parse_grounding_response(P.get(base + "_tl", ""))
            sp = [a, b] if a is not None and b is not None else None
        sp = sp or [-100.0, -100.0]
        v = [get_tIoU(g, sp) for g in s["nextgqa"]["spans"]]
        iou, iop = max(0, max(x[0] for x in v)), max(0, max(x[1] for x in v))
        out[base] = (iop, float(iop >= 0.5), iou, float(iou >= 0.5), float(sp[0] == 0))
    return out


def main():
    ref = per_q(sys.argv[1])
    names = ["mIoP", "IoP@0.5", "mIoU", "IoU@0.5", "start0"]
    for run in sys.argv[1:]:
        m = per_q(run)
        ks = sorted(m)
        line = " ".join(f"{n} {100 * sum(m[k][i] for k in ks) / len(ks):.1f}" for i, n in enumerate(names))
        if run != sys.argv[1]:
            rng = random.Random(0)
            for i, n in ((0, "dmIoP"), (2, "dmIoU")):
                d = [m[k][i] - ref[k][i] for k in ks]
                bs = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
                line += f" | {n} {100 * sum(d) / len(d):+.1f} [{bs[50]:+.1f}, {bs[1949]:+.1f}]"
        print(f"{run.rstrip('/').split('/')[-1]:24s} {line}")


if __name__ == "__main__":
    main()
