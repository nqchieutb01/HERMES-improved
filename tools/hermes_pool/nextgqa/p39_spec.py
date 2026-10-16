"""Phase 39: NExT-GQA test subset (prepare.py: ~1,000 questions on the videos drawn, two samples per question —
VideoMind's multiple-choice prompt and TimeLens' grounding prompt with the question as query). Qwen3-VL-8B, 64 uniform
frames at native resolution (NExT-QA videos are ~40 s, 320x240 or 640x480; retrieval is off on such short videos),
greedy decoding, no repetition penalty. Scored with the official NExT-GQA evaluation (score.py).
t8-*: TimeLens-8B (Qwen3-VL-8B tuned for grounding; results under timelens_8b/), same settings; frames pre-resized as
on S-EMBER (a no-op for these low-resolution videos).
Usage: python3 p39_spec.py [rest|t8] > p39.json | p39_rest.json | p39_t8.json   (rest: all but the Qwen3-VL unpruned
check run; t8: TimeLens-8B Cover / random 10%)
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/nextgqa"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/nextgqa/"
BASE = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", f"dataset.annotation={D}/nextgqa_test_subset.jsonl",
        f"dataset.video_root={D}/videos", "dataset.max_videos=100000", "dataset.question_categories=[moment_retrieval]",
        "dataset.grounding_prompt=raw", "run.uniform_num_frames=64", "run.keep_time_tokens=surviving",
        "run.max_new_tokens=256", "run.repetition_penalty=1.0", "run.prefetch_videos=4"]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
RUNS = {  # name: (overrides, batch size)
    "unpruned": (["run.prune_score=random", "run.offline_keep_ratio=1.0"], 2),
    "random30": (["run.prune_score=random", "run.offline_keep_ratio=0.3"], 8),
    "semvid30": (["run.prune_score=semvid", "run.offline_keep_ratio=0.3"], 8),
    "cover30": (["run.prune_score=core_topk", "run.offline_keep_ratio=0.3"], 8),
    "random10": (["run.prune_score=random", "run.offline_keep_ratio=0.1"], 8),
    "cover10": (["run.prune_score=core_topk", "run.offline_keep_ratio=0.1"], 8),
    "t8-unpruned": (["model.model_path=/nfs-stor/chieu.nguyen/models/TimeLens-8B", "run.frame_max_pixels=1572864",
                     "run.frame_min_pixels=0", "run.prune_score=random", "run.offline_keep_ratio=1.0"], 2),
    **{f"t8-{n}": (["model.model_path=/nfs-stor/chieu.nguyen/models/TimeLens-8B", "run.frame_max_pixels=1572864",
                    "run.frame_min_pixels=0", f"run.prune_score={score}", "run.offline_keep_ratio=0.1"], 8)
       for n, score in (("cover10", "core_topk"), ("random10", "random"))},
}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]
runs = [{"name": f"p39-{tag}", "code": CODE, "chunks": 8, "est_min": 30, "priority": 10, "env": ENV,
         "overrides": BASE + over + [f"run.batch_size={bs}", "paths.save_dir=" + (
             f"{R.replace('qwen3_vl_8b', 'timelens_8b')}{tag[3:]}" if tag.startswith("t8-") else f"{R}{tag}")]}
        for tag, (over, bs) in RUNS.items()]
if __import__("sys").argv[1:] == ["rest"]:
    runs = [r for r in runs if r["name"] != "p39-unpruned"]
if __import__("sys").argv[1:] == ["t8"]:  # TimeLens-8B pruned, after its unpruned check
    runs = [r for r in runs if r["name"] in ("p39-t8-cover10", "p39-t8-random10")]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs}, indent=1))
