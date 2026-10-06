"""Phase 21d: FastVID (NeurIPS 2025; inference/fastvid.py) as a recent token-pruning baseline, with and without PCD.

FastVID 10% (input-level: dynamic temporal segmentation, salient + density-peak anchors, anchor-centric merging,
authors' Qwen2.5-VL defaults), timeline prompt, baseline and final method (adaptive alpha_max 1, time scope).
Per-frame retention is recorded for the coverage analysis (logs/phase21/pruning_link.py).
Usage: cd logs/phase21 && python3 make_spec_d.py > d.json
"""
import json

from make_spec import SLOTS, run

RUNS = []
for method in ("base", "final"):
    r = run(0.1, method, "timeline")
    r["name"] = f"p21d-fastvid10-{method}-timeline"
    r["overrides"] = [o.replace("run.prune_score=random", "run.prune_score=fastvid") for o in r["overrides"][:-1]] + [
        "run.retention_snapshot=true",
        r["overrides"][-1].replace("-p21-random10-", "-p21d-fastvid10-")]
    RUNS.append(r)

if __name__ == "__main__":
    print(json.dumps({"slots": [dict(s, max_jobs=6 if s["partition"] == "long" else 2) for s in SLOTS],
                      "runs": RUNS}, indent=1))
