"""Phase 21b: why events-format answers run on, and two fixes (no output post-processing).

Every run-on answer of the events prompt stopped at the 384-token limit (an event line costs ~25 tokens; most are
counting questions with many occurrences), and under PCD 80% of those lists split one continuing action into
back-to-back fixed-length events. Fixes: a 768-token budget, and the events_merged prompt (consecutive moments of one
continuing action are one event). Baseline and final method at random 10%.
Usage: cd logs/phase21 && python3 make_spec_b.py > b.json
"""
import json

from make_spec import SLOTS, run

RUNS = []
for prompt in ("events", "events_merged"):
    for method in ("base", "final"):
        r = run(0.1, method, prompt)
        r["name"] = r["name"].replace("p21-", "p21b-") + "-768"
        r["overrides"] = [o.replace("run.max_new_tokens=384", "run.max_new_tokens=768") for o in r["overrides"][:-1]] + [
            r["overrides"][-1].replace("-p21-", "-p21b-").replace("-v300", "-768-v300")]
        RUNS.append(r)

if __name__ == "__main__":
    print(json.dumps({"slots": [dict(s, max_jobs=4 if s["partition"] == "long" else 1) for s in SLOTS],
                      "runs": RUNS}, indent=1))
