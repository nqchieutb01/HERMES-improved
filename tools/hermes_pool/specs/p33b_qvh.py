"""Phase 33b (QVHighlights, same subset as phase 33): PCD without the answer reserve (the reserve's effect on this
dataset) and the final method at 5% and 20% of the visual tokens. Usage: python3 p33b_qvh.py > p33b_qvh.json
"""
import importlib.util
import json

_spec = importlib.util.spec_from_file_location("p33", "/home/chieu.nguyen/hermes_pool/specs/p33_qvh.py")
p33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p33)

PCD = [o for o in p33.PCD_AR if not o.startswith("run.answer_reserve")]
RC = ["run.prune_score=core_topk", "run.retrieval=frames_cont"]
RANDOM = ["run.prune_score=random", "run.prefetch_videos=4"]
CONFIGS = {  # name: (overrides, est_min, priority)
    "random10-pcd": (p33.K10 + p33.BS8 + RANDOM + PCD, 40, 41),
    "rc64-qtopk10-pcd": (p33.K10 + p33.BS8 + RC + PCD, 40, 41),
    "rc64-qtopk5-pcd-ar64": (["run.offline_keep_ratio=0.05"] + p33.BS8 + RC + p33.PCD_AR, 40, 42),
    "rc64-qtopk20-pcd-ar64": (["run.offline_keep_ratio=0.2"] + p33.BS8 + RC + p33.PCD_AR, 40, 42),
}
runs = [{"name": f"p33b-{tag}", "code": p33.CODE, "chunks": 8, "est_min": est, "priority": prio, "env": p33.ENV,
         "overrides": p33.BASE + over + [f"paths.save_dir={p33.R}{tag}"]} for tag, (over, est, prio) in CONFIGS.items()]

if __name__ == "__main__":
    print(json.dumps({"slots": p33.SLOTS, "runs": runs}, indent=1))
