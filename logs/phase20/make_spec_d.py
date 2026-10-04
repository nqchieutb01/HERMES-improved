"""Phase 20d: confidence-adaptive PCD restricted to time values (alpha_max 1, scope time) on every remaining memory.

Phase 20/20c: adaptive + time scope gave the best unpruned result (Acc. 21.5, mIoU 32.7, GQ@0.5 13.3 vs PCD 19.8 /
31.5 / 11.2) and matched PCD at random 10%; time scope alone was best for HERMES 10% and streaming 4k (phase 19).
This round completes the combination on the other memories so the final method has one configuration everywhere.
Usage: cd logs/phase20 && python3 make_spec_d.py > d.json
"""
import json

from make_spec import SLOTS, run
from make_spec_c import stream


def scoped_stream(kv):
    r = stream(kv, 1.0)
    r["name"] += "-scopetime"
    r["overrides"] = [o.replace("adapt1.0-timeline", "adapt1.0-scopetime-timeline") if o.startswith("paths.save_dir")
                      else o for o in r["overrides"] if not o.startswith("run.contrastive_scope")]
    r["overrides"].insert(-1, "run.contrastive_scope=time")
    return r


RUNS = [run("random", 0.05, 1.0, "time"), run("random", 0.25, 1.0, "time"), run("hermes", 0.1, 1.0, "time"),
        run("stratified", 0.1, 1.0, "time"), scoped_stream(4000), scoped_stream(6000)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
