"""Phase 20: the first time value the model writes, and what each contrast rule would choose there.

Reads the decode trace of a greedy run (rule 'against', alpha 0: the scores equal log p(. | M)), takes the first step that
writes a time value ("Seen: <t>" or "Time: [<t>") and, for each rule and strength, simulates the choice from the logged
top-20 tokens under the real memory M and the blind prior C (plausibility cut-off beta). Reports, for early-evidence
questions (gold start < 5% of the question time) and the rest, the share of choices that are "0" — correct for early
questions, prior leakage for the others — and the separability (AUC) of candidate signals for a supported "0".
Run from the repo root: python3 logs/phase20/first_time_token.py [trace run tag]
"""
import glob
import json
import math
import sys

sys.path.insert(0, "logs/phase15")
from diagnose_gpu import run_dir  # noqa: E402

ANNO = "/nfs-stor/chieu.nguyen/s-ember/sember_grounding.jsonl"

TAG = sys.argv[1] if len(sys.argv) > 1 else "random10-cd-blind-against0.0-timeline"
BETA = 0.1


def first_step(steps):
    for s in steps:
        # The first digit of the first time value (the step after "Seen:" may write only the space).
        if s["prefix"].rstrip().endswith(("Seen:", "Time: [", "Time:[")) and s["chosen"].strip().isdigit():
            return s
    return steps[0] if steps else None


def choose(top, rule, a):
    best, arg = -math.inf, None
    lmax = max(lp for _, lp, _ in top)
    for t, lp, ln in top:
        if lp < lmax + math.log(BETA):
            continue
        if rule == "pmi":
            s = (1 + a) * lp - a * ln
        elif rule == "adaptive":
            b = a * (1 - math.exp(lmax))
            s = (1 + b) * lp - b * ln
        else:
            s = lp - a * max(0.0, ln - lp)
        if s > best:
            best, arg = s, t
    return arg.strip()


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1)


def main():
    rows = {}
    for l in open(ANNO):
        d = json.loads(l)
        rows[d["question_id"]] = {"_gs": float(d["answer_start_time"]), "_qt": float(d["question_time"])}
    traces = {}
    for f in glob.glob(f"{run_dir(TAG)}/cdtrace-*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            s = first_step(d["steps"])
            if s:
                traces[d["question_id"]] = s
    groups = {"early": [], "later": []}
    for q, s in traces.items():
        r = rows[q]
        groups["early" if r["_gs"] < 0.05 * r["_qt"] else "later"].append(s)
    print(f"{TAG}: first time token for {len(traces)} questions "
          f"(early {len(groups['early'])}, later {len(groups['later'])})")
    print(f"{'rule':10s} {'a':>4s} | first token '0': early (want high) | later (leak, want low)")
    for rule, alphas in (("pmi", (0, 0.25, 0.5, 1, 2)), ("adaptive", (1, 2, 4)), ("against", (0.5, 1, 2, 4, 8))):
        for a in alphas:
            z = {g: 100 * sum(choose(s["top"], rule, a) == "0" for s in v) / max(len(v), 1) for g, v in groups.items()}
            print(f"{rule:10s} {a:4g} | {z['early']:5.1f} | {z['later']:5.1f}")

    def zero(s):
        for t, lp, ln in s["top"]:
            if t.strip() == "0":
                return lp, ln
        return None

    print("\nsignals for '0' (AUC early vs later; > 0.5 = higher for early-evidence questions):")
    sig = {"p_M(0)": lambda lp, ln: lp, "p_C(0)": lambda lp, ln: ln, "log LR(0) = log p_M - log p_C": lambda lp, ln: lp - ln}
    for name, f in sig.items():
        v = {g: [f(*zero(s)) for s in vs if zero(s)] for g, vs in groups.items()}
        med = {g: sorted(x)[len(x) // 2] if x else float("nan") for g, x in v.items()}
        print(f"  {name:32s} AUC {auc(v['early'], v['later']):.3f} | median early {med['early']:+.2f} later {med['later']:+.2f}")


if __name__ == "__main__":
    main()
