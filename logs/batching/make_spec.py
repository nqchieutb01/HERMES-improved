"""Full-dataset check of batched answering: two existing batch-size-1 configurations rerun with batch size 8.

Usage: cd logs/batching && python3 make_spec.py > full.json
"""
import json
import sys

sys.path.insert(0, "../phase20")
from make_spec import SLOTS, run  # noqa: E402

pcd = run("random", 0.1, 1.0, "time")  # random10-cd-blind-adapt1.0-scopetime-timeline
base = run("random", 0.1, 1.0)
base["overrides"] = [o for o in base["overrides"] if not o.startswith(("run.contrastive", "paths.save_dir"))] + [
    "paths.save_dir=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/"
    "uniform-n64-time-count-location-offline-random-keep0.1-timeline-v300-native-time"]
RUNS = []
for r, name in ((base, "g-random10-timeline-bs8"), (pcd, "g-random10-cd-blind-adapt1.0-scopetime-timeline-bs8")):
    r = dict(r, name=name)
    *rest, save = r["overrides"]
    r["overrides"] = rest + ["run.batch_size=8", save.replace("-v300-native-time", "-bs8-v300-native-time")]
    RUNS.append(r)

if __name__ == "__main__":
    slots = [dict(s, max_jobs=2 if s["partition"] == "long" else 0) for s in SLOTS]
    print(json.dumps({"slots": slots, "runs": RUNS}, indent=1))
