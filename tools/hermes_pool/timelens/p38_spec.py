"""Phase 38 on the TimeLens-Bench QVHighlights subset (513 queries), protocol input as in phase 37.

a: PCD under our timeline prompt (Qwen3-VL-8B). Answers list "Seen:" moments, then "Answer:" and "Time: [s, e]";
   scored on the final Time line (score.py --time-line), the one deviation from the TimeLens protocol. Questions are
   phrased "When does the following happen in the video: '<query>'?" (qvh_tl_subset_timeline.jsonl, same ids).
b: Cover on TimeLens-8B (fine-tuned Qwen3-VL-8B, the strongest open VTG model), TimeLens prompt and metric, same
   budgets as phase 37. TimeLens-script runs for b are queued by launch_p38.sh.
Usage: python3 p38_spec.py a|b > p38<x>.json
"""
import importlib.util
import json
import sys

_s = importlib.util.spec_from_file_location("p37", "/home/chieu.nguyen/hermes_pool/timelens/p37_spec.py")
p37 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p37)

D = p37.D
R8 = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/timelens_qvh/"
RT = "/home/chieu.nguyen/HERMES/results/timelens_8b/timelens_qvh/"
TIMELENS_8B = "/nfs-stor/chieu.nguyen/models/TimeLens-8B"
TIMELINE = [o.replace("qvh_tl_subset.jsonl", "qvh_tl_subset_timeline.jsonl").replace(
    "dataset.grounding_prompt=timelens", "dataset.grounding_prompt=timeline") for o in p37.BASE]
AR = ["run.answer_reserve=64"]
P300 = p37.PROTOCOL
COVER, RANDOM = ["run.prune_score=core_topk"], ["run.prune_score=random"]
K31 = ["run.offline_keep_ratio=0.31", "run.prefetch_videos=4"]

A = {  # name: (overrides, batch size, est_min)
    "tlp-unpruned": (TIMELINE + P300 + RANDOM + ["run.offline_keep_ratio=1.0", "run.prefetch_videos=4"], 1, 50),
    "tlp-random31": (TIMELINE + P300 + RANDOM + K31, 8, 35),
    "tlp-random31-pcd-ar64": (TIMELINE + P300 + RANDOM + K31 + p37.PCD + AR, 8, 40),
    "tlp-cover31": (TIMELINE + P300 + COVER + K31, 8, 35),
    "tlp-cover31-pcd-ar64": (TIMELINE + P300 + COVER + K31 + p37.PCD + AR, 8, 40),
}
B = {
    "bridge-unpruned": (p37.A["bridge-unpruned"][0], 1, 45),
    "p300-random31": (p37.A["p300-random31"][0], 8, 30),
    "p300-cover31": (p37.A["p300-cover31"][0], 8, 30),
    "rc64-cover23": (p37.B["rc64-cover23"][0], 8, 20),
}


def runs(group, prefix, base, root, extra, prio):
    return [{"name": f"{prefix}-{tag}", "code": p37.CODE, "chunks": 8, "est_min": est, "priority": prio,
             "env": p37.ENV, "overrides": base + over + extra + [f"run.batch_size={bs}", f"paths.save_dir={root}{tag}"]}
            for tag, (over, bs, est) in group.items()]


if __name__ == "__main__":
    if sys.argv[1] == "a":
        out = runs(A, "p38", [], R8, [], 23)
    else:
        out = runs(B, "p38t8", p37.BASE, RT, [f"model.model_path={TIMELENS_8B}"], 24)
    print(json.dumps({"slots": p37.SLOTS, "runs": out}, indent=1))
