"""Phase 42: NExT-GQA open-ended (prepare_open.py; same 1,005 test questions, no options), 64 uniform frames, batch 8
for pruned runs and 2 for unpruned ones (8 runs out of memory), as the multiple-choice rows; both use the batched
decoder, which the answer reserve needs.
- oe (two queries, as the multiple-choice protocol minus options): "<question>? Answer the question using a single word
  or phrase." + TimeLens grounding query; grounding_prompt=raw, 256 new tokens, repetition penalty 1.0.
- tl (joint, S-EMBER timeline prompt): one reply with Seen / Answer / Time; 384 new tokens, repetition penalty 1.1 (as
  the S-EMBER rows, where the method was developed).
Retrieval is not used: 64 frames over ~40 s videos are < 1 s apart (the coverage gate would switch it off), so the
method rows are Cover (+ PCD + reserve). Answers judged by judge_open.py (TOGA / Video-ChatGPT yes-no), scored by
score.py --open oe|tl.
Sets: check = Qwen3-VL-8B unpruned in both protocols; sweep = the pruned Qwen3-VL rows; t8 = TimeLens-8B rows.
Usage: python3 p42_spec.py check|sweep|t8 > p42_<set>.json
"""
import json
import sys

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/nextgqa"
RES = "/home/chieu.nguyen/HERMES/results/"
T8 = ["model.model_path=/nfs-stor/chieu.nguyen/models/TimeLens-8B", "run.frame_max_pixels=1572864", "run.frame_min_pixels=0"]
COMMON = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", f"dataset.video_root={D}/videos",
          "dataset.max_videos=100000", "dataset.question_categories=[moment_retrieval]", "run.uniform_num_frames=64",
          "run.keep_time_tokens=surviving", "run.prefetch_videos=4"]
PROTO = {
    "oe": [f"dataset.annotation={D}/nextgqa_test_oe.jsonl", "dataset.grounding_prompt=raw", "run.max_new_tokens=256",
           "run.repetition_penalty=1.0"],
    "tl": [f"dataset.annotation={D}/nextgqa_test_tl.jsonl", "dataset.grounding_prompt=timeline",
           "run.max_new_tokens=384", "run.repetition_penalty=1.1"],
}
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time", "run.answer_reserve=64"]
M = {
    "unpruned": ["run.prune_score=random", "run.offline_keep_ratio=1.0"],
    "random10": ["run.prune_score=random", "run.offline_keep_ratio=0.1"],
    "cover10": ["run.prune_score=core_topk", "run.offline_keep_ratio=0.1"],
    "random30": ["run.prune_score=random", "run.offline_keep_ratio=0.3"],
    "cover30": ["run.prune_score=core_topk", "run.offline_keep_ratio=0.3"],
    "semvid10": ["run.prune_score=semvid", "run.offline_keep_ratio=0.1"],
    "semvid30": ["run.prune_score=semvid", "run.offline_keep_ratio=0.3"],
}


def run(model, proto, tag, over, prio):
    mt = "timelens_8b" if model == "t8" else "qwen3_vl_8b"
    bs = 2 if tag.startswith("unpruned") else 8  # unpruned at batch 8 runs out of memory (as the MCQ rows: 2 / 8)
    return {"name": f"p42-{'t8-' if model == 't8' else ''}{proto}-{tag}", "code": CODE, "chunks": 8, "est_min": 30,
            "priority": prio, "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"},
            "overrides": COMMON + PROTO[proto] + (T8 if model == "t8" else []) + over
            + [f"run.batch_size={bs}", f"paths.save_dir={RES}{mt}/nextgqa_open/{proto}-{tag}"]}


SWEEP_OE = ["random10", "cover10", "random30", "cover30", "semvid10", "semvid30"]
SWEEP_TL = [(t, M[t]) for t in ["random10", "cover10", "random30", "cover30", "semvid10", "semvid30"]] + [
    ("unpruned-pcd-ar64", M["unpruned"] + PCD), ("random10-pcd-ar64", M["random10"] + PCD),
    ("cover10-pcd-ar64", M["cover10"] + PCD), ("random30-pcd-ar64", M["random30"] + PCD),
    ("cover30-pcd-ar64", M["cover30"] + PCD), ("semvid10-pcd-ar64", M["semvid10"] + PCD)]
SETS = {
    "check": [run("q", "oe", "unpruned", M["unpruned"], 4), run("q", "tl", "unpruned", M["unpruned"], 4)],
    "sweep": [run("q", "tl", t, o, 5) for t, o in SWEEP_TL] + [run("q", "oe", t, M[t], 6) for t in SWEEP_OE],
    "t8": [run("t8", p, t, o, 7) for p, t, o in [("oe", "unpruned", M["unpruned"]), ("tl", "unpruned", M["unpruned"]),
                                                 ("tl", "random10", M["random10"]), ("tl", "cover10", M["cover10"]),
                                                 ("tl", "random10-pcd-ar64", M["random10"] + PCD),
                                                 ("tl", "cover10-pcd-ar64", M["cover10"] + PCD),
                                                 ("tl", "unpruned-pcd-ar64", M["unpruned"] + PCD),
                                                 ("oe", "random10", M["random10"]), ("oe", "cover10", M["cover10"])]],
}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": SETS[sys.argv[1]]}, indent=1))
