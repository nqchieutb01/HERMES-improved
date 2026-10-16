"""Phase 37: our pipeline on the TimeLens-Bench QVHighlights subset (shard 01: 513 queries, 506 videos, ~150 s,
native 534x300), TimeLens protocol: their prompt, greedy decoding with no repetition penalty, 512 new tokens, their
metric (score.py). Every pruned run keeps the same number of visual tokens B as a TimeLens-script baseline:
  protocol input = 2 fps (300 frames) resized as qwen_vl_utils does under total_tokens 14336 / min_tokens 64
  (224x416, 91 tokens per frame pair, 13,650 in total);
  native = 288x544 (153 tokens per pair): 128 frames = 9,792, 64 frames = 4,896.
B ~ 4.2k: protocol x 0.31 = 4,231; 128 native x 0.43 = 4,210.   B ~ 1.1k: protocol x 0.082 = 1,119; 64 native x 0.23 = 1,126.
p37a: runs without retrieval; p37b: retrieval runs (need the SigLIP index). Usage: python3 p37_spec.py a|b > p37<x>.json
"""
import json
import sys

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/timelens-bench"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/timelens_qvh/"
BASE = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", f"dataset.annotation={D}/qvh_tl_subset.jsonl",
        f"dataset.video_root={D}/videos_flat", "dataset.max_videos=1000", "dataset.question_categories=[moment_retrieval]",
        "dataset.grounding_prompt=timelens", "run.keep_time_tokens=surviving", "run.max_new_tokens=512",
        "run.repetition_penalty=1.0"]
PROTOCOL = ["run.uniform_num_frames=300", "run.frame_max_pixels=97867", "run.frame_min_pixels=65536"]
NATIVE = ["run.frame_max_pixels=786432", "run.frame_min_pixels=16384"]  # VIDEO_MAX_PIXELS: native 288x544
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time"]
RANDOM = ["run.prune_score=random"]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2", "HERMES_RETRIEVAL_INDEX": f"{D}/siglip2_1fps"}

A = {  # name: (overrides, batch size, est_min)
    "bridge-unpruned": (PROTOCOL + RANDOM + ["run.offline_keep_ratio=1.0", "run.prefetch_videos=4"], 1, 45),
    "p300-random31": (PROTOCOL + RANDOM + ["run.offline_keep_ratio=0.31", "run.prefetch_videos=4"], 8, 30),
    "p300-semvid31": (PROTOCOL + ["run.prune_score=semvid", "run.offline_keep_ratio=0.31", "run.prefetch_videos=4"], 8, 30),
    "p300-cover31": (PROTOCOL + ["run.prune_score=core_topk", "run.offline_keep_ratio=0.31", "run.prefetch_videos=4"], 8, 30),
    "p300-cover31-pcd": (PROTOCOL + ["run.prune_score=core_topk", "run.offline_keep_ratio=0.31",
                                     "run.prefetch_videos=4"] + PCD, 8, 35),
    "p300-random8": (PROTOCOL + RANDOM + ["run.offline_keep_ratio=0.082", "run.prefetch_videos=4"], 8, 30),
}
B = {
    "rc128-random43": (NATIVE + RANDOM + ["run.uniform_num_frames=128", "run.retrieval=frames_cont",
                                          "run.offline_keep_ratio=0.43"], 8, 25),
    "rc128-cover43": (NATIVE + ["run.prune_score=core_topk", "run.uniform_num_frames=128", "run.retrieval=frames_cont",
                                "run.offline_keep_ratio=0.43"], 8, 25),
    "rc128-cover43-pcd": (NATIVE + ["run.prune_score=core_topk", "run.uniform_num_frames=128",
                                    "run.retrieval=frames_cont", "run.offline_keep_ratio=0.43"] + PCD, 8, 30),
    "rc64-random23": (NATIVE + RANDOM + ["run.uniform_num_frames=64", "run.retrieval=frames_cont",
                                         "run.offline_keep_ratio=0.23"], 8, 20),
    "rc64-cover23": (NATIVE + ["run.prune_score=core_topk", "run.uniform_num_frames=64", "run.retrieval=frames_cont",
                               "run.offline_keep_ratio=0.23"], 8, 20),
    "rc64-cover23-pcd": (NATIVE + ["run.prune_score=core_topk", "run.uniform_num_frames=64", "run.retrieval=frames_cont",
                                   "run.offline_keep_ratio=0.23"] + PCD, 8, 25),
}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]


def runs(group, prio):
    return [{"name": f"p37-{tag}", "code": CODE, "chunks": 8, "est_min": est, "priority": prio, "env": ENV,
             "overrides": BASE + over + [f"run.batch_size={bs}", f"paths.save_dir={R}{tag}"]}
            for tag, (over, bs, est) in group.items()]


if __name__ == "__main__":
    which = sys.argv[1]
    print(json.dumps({"slots": SLOTS, "runs": runs(A, 21) if which == "a" else runs(B, 22)}, indent=1))
