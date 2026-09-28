"""Answer-aware grounding metrics for every S-EMBER grounding run (no GPU).

Per run: MCQ accuracy of the matching MCQ run (same run name), grounding mIoU and R@0.5, answer accuracy
(LLM judge, from answer_judgments.jsonl written by judge_grounding.py; and rule-based for duration and
counting questions), and the joint metric: a question counts only if its answer is correct AND its
interval has IoU >= 0.5 (GQ@0.5, the official S-EMBER joint metric) or >= 0.3 (GQ@0.3). Acc is the
judge accuracy with the official S-EMBER judge prompt. Rows are sorted by GQ@0.5.
Answer correctness comes from the judge when answer_judgments.jsonl exists, otherwise from the rule-based
check (duration and counting questions only). Run from the repo root: python3 logs/phase11/joint_metrics.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_answer as ra  # noqa: E402

B = "results/qwen3_vl_8b"


def truthy(v):
    return v in (True, "True", "true", 1, "1.0")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def short(tag):
    for a, b in (("time-count-location-", ""), ("-attention_weighted-native-time-keepsurv", ""),
                 ("-native-time", ""), ("-v300", ""), ("uniform-n", "uni")):
        tag = tag.replace(a, b)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="logs/phase11/runs.txt")
    args = ap.parse_args()
    out = []
    for run in (l.strip() for l in open(args.runs) if l.strip()):
        tag = os.path.basename(run)
        rows = load_jsonl(f"results/{run}/sember_grounding_scored.jsonl")
        judged = load_jsonl(f"results/{run}/answer_judgments.jsonl")
        judge = {j["question_id"]: j["correct"] for j in judged} if judged else None
        mcq = load_jsonl(f"{B}/sember_mcq/{tag}/sember_mcq_scored.jsonl")
        n = len(rows)
        iou = [float(r["temporal_iou"] or 0) for r in rows]
        rule = [ra.rule_correct(r) for r in rows]
        covered = [x for x in rule if x is not None]
        # Answer correctness per question: judge when available, else the rule (None -> unknown).
        correct = [judge.get(r["question_id"]) if judge else x for r, x in zip(rows, rule)]
        known = [c is not None for c in correct]
        joint3 = [bool(c) and i >= 0.3 for c, i in zip(correct, iou)]
        joint5 = [bool(c) and i >= 0.5 for c, i in zip(correct, iou)]
        denom = n if judge else max(1, sum(known))
        out.append({
            "run": short(tag),
            "mcq": 100 * sum(truthy(r["is_correct"]) for r in mcq) / len(mcq) if mcq else None,
            "miou": 100 * sum(iou) / n,
            "r05": 100 * sum(truthy(r["recall_at_1_iou_0.5"]) for r in rows) / n,
            "judge": 100 * sum(bool(judge.get(r["question_id"])) for r in rows) / n if judge else None,
            "rule": 100 * sum(covered) / len(covered) if covered else None,
            "j3": 100 * sum(j for j, k in zip(joint3, known) if k) / denom,
            "j5": 100 * sum(j for j, k in zip(joint5, known) if k) / denom,
            "basis": "judge, all 475" if judge else f"rule, {sum(known)} dur/count",
        })
    out.sort(key=lambda d: -d["j5"])
    f = lambda v: "  --" if v is None else f"{v:4.1f}"
    print(f"{'run':52s} {'MCQ':>5s} {'mIoU':>5s} {'R@.5':>5s} {'Acc':>5s} {'AnsR':>5s} {'GQ@.3':>6s} {'GQ@.5':>6s}  basis")
    for d in out:
        print(f"{d['run'][:52]:52s} {f(d['mcq']):>5s} {f(d['miou']):>5s} {f(d['r05']):>5s} {f(d['judge']):>5s} "
              f"{f(d['rule']):>5s} {f(d['j3']):>6s} {f(d['j5']):>6s}  {d['basis']}")


if __name__ == "__main__":
    main()
