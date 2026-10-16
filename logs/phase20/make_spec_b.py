"""Phase 20b: evidence-against contrast, log p_M - alpha * max(0, log p_blind - log p_M), with decode traces.

Constant and confidence-adaptive PMI contrast both lose correct early starts (random 10%: 79% -> 61% of early-evidence
questions start in the first 5%), because the PMI term rewards any time the blind prior finds improbable. The
evidence-against rule only discounts tokens the video argues against. alpha = 0 is greedy decoding with the trace (the
reference for offline analysis of the first time token). Usage: python3 logs/phase20/make_spec_b.py > logs/phase20/b.json
"""
import json

from make_spec import SLOTS, run

RUNS = []
for score, keep, a in (("random", 0.1, 0.0), ("random", 0.1, 1.0), ("random", 0.1, 2.0), ("random", 0.1, 4.0),
                       ("random", 1.0, 2.0)):
    r = run(score, keep, a)
    tag = r["name"].replace(f"adapt{a}", f"against{a}")
    r["name"] = tag
    r["overrides"] = [o for o in r["overrides"] if not o.startswith(("run.contrastive_adaptive", "paths.save_dir"))] + [
        "run.contrastive_rule=against", "run.contrastive_trace=true",
        f"paths.save_dir=" + r["overrides"][-1].split("=", 1)[1].replace(f"adapt{a}", f"against{a}")]
    RUNS.append(r)

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
