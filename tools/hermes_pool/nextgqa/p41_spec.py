"""Phase 41: NExT-GQA joint protocol (prepare_joint.py: LOVER's prompt, one reply with <answer> span and <choice>),
same 1,005 questions and settings as the phase-39 unpruned runs (64 uniform frames, greedy, no repetition penalty),
max_new_tokens 1,024 for the <think> part. Tests whether our two-query protocol (VideoMind MCQ + TimeLens grounding
query) explains the Acc&GQA above published zero-shot rows. Scored with score.py --joint.
Usage: python3 p41_spec.py > p41.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/nextgqa"
RES = "/home/chieu.nguyen/HERMES/results/"
BASE = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", f"dataset.annotation={D}/nextgqa_test_joint.jsonl",
        f"dataset.video_root={D}/videos", "dataset.max_videos=100000", "dataset.question_categories=[moment_retrieval]",
        "dataset.grounding_prompt=raw", "run.uniform_num_frames=64", "run.keep_time_tokens=surviving",
        "run.max_new_tokens=1024", "run.repetition_penalty=1.0", "run.prefetch_videos=4",
        "run.prune_score=random", "run.offline_keep_ratio=1.0", "run.batch_size=2"]
T8 = ["model.model_path=/nfs-stor/chieu.nguyen/models/TimeLens-8B", "run.frame_max_pixels=1572864",
      "run.frame_min_pixels=0"]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]
runs = [{"name": f"p41-{tag}", "code": CODE, "chunks": 8, "est_min": 30, "priority": 5, "env": ENV,
         "overrides": BASE + extra + [f"paths.save_dir={RES}{mt}/nextgqa/joint-unpruned"]}
        for tag, mt, extra in (("joint-unpruned", "qwen3_vl_8b", []), ("t8-joint-unpruned", "timelens_8b", T8))]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs}, indent=1))
