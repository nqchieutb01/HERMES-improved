"""Phase 30: token budget sweep on S-EMBER (300 videos, 475 questions), 64 frames, 5% and 20% of visual tokens.

Random, SemVID (input-level, ECCV'26) and Retrieve + Cover (retrieval frames + question top-k per frame), each with
and without final PCD. Same settings as phase 25 (the 10% points are p25 / offline-random-keep0.1 / random10-cd).
Usage: python3 p30_budgets.py > p30_budgets.json
"""
import json

R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform", "run.uniform_num_frames=64",
        "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
        "run.batch_size=8", "run.retention_snapshot=true"]
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time"]
METHODS = {"random": ["run.prune_score=random", "run.prefetch_videos=4"],
           "u64-semvid": ["run.prune_score=semvid", "run.prefetch_videos=4"],
           "rc64-qtopk": ["run.prune_score=core_topk", "run.retrieval=frames_cont"]}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]
runs = []
for keep, tag in ((0.05, "5"), (0.2, "20")):
    for m, over in METHODS.items():
        for pcd in (False, True):
            name = f"p30-{m}{tag}{'-pcd' if pcd else ''}"
            runs.append({"name": name, "code": "/home/chieu.nguyen/HERMES", "chunks": 4, "est_min": 50,
                         "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"},
                         "overrides": BASE + over + [f"run.offline_keep_ratio={keep}"] + (PCD if pcd else [])
                         + [f"paths.save_dir={R}{name}-timeline-v300-native-time"]})
print(json.dumps({"slots": SLOTS, "priority": 40, "runs": runs}, indent=1))
