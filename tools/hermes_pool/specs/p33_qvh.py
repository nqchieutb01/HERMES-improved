"""Phase 33: token selection on QVHighlights (val, 500 single-window queries, ~150 s videos), 64 frames, 10% of tokens.

Same lineup as phase 28 (Charades-STA) with the final decoding (PCD + answer reserve 64). Uniform frames are ~2.3 s
apart here, so retrieval frames are active (the retrieval_min_gap gate would not switch them off).
Code: worktree branch worktree-p31-gate-pcdfix. Usage: python3 p33_qvh.py > p33_qvh.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/qvhighlights"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/qvhighlights/"
BASE = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", f"dataset.annotation={D}/qvh_val_single_500.jsonl",
        f"dataset.video_root={D}/videos_flat", "dataset.max_videos=1000",
        "dataset.question_categories=[moment_retrieval]", "run.uniform_num_frames=64",
        "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384"]
PCD_AR = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
          "run.contrastive_scope=time", "run.answer_reserve=64"]
K10, BS8 = ["run.offline_keep_ratio=0.1"], ["run.batch_size=8"]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2", "HERMES_RETRIEVAL_INDEX": f"{D}/siglip2_1fps"}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]

CONFIGS = {  # name: (overrides, est_min per chunk)
    "unpruned": (["run.offline_keep_ratio=1.0", "run.prune_score=random", "run.prefetch_videos=4",
                  "run.batch_size=1"], 70),
    "random10": (K10 + BS8 + ["run.prune_score=random", "run.prefetch_videos=4"], 35),
    "random10-pcd-ar64": (K10 + BS8 + ["run.prune_score=random", "run.prefetch_videos=4"] + PCD_AR, 40),
    "fastvid10": (K10 + BS8 + ["run.prune_score=fastvid", "run.prefetch_videos=4"], 35),
    "vidcom210": (K10 + BS8 + ["run.prune_score=vidcom2", "run.prefetch_videos=4"], 35),
    "flashvid10": (K10 + BS8 + ["run.prune_score=flashvid", "run.prefetch_videos=4"], 35),
    "semvid10": (K10 + BS8 + ["run.prune_score=semvid", "run.prefetch_videos=4"], 35),
    "semvid10-pcd-ar64": (K10 + BS8 + ["run.prune_score=semvid", "run.prefetch_videos=4"] + PCD_AR, 40),
    "u64-qtopk10": (K10 + BS8 + ["run.prune_score=core_topk", "run.prefetch_videos=4"], 35),
    "rc64-rand10": (K10 + BS8 + ["run.prune_score=random", "run.retrieval=frames_cont"], 35),
    "rc64-qtopk10": (K10 + BS8 + ["run.prune_score=core_topk", "run.retrieval=frames_cont"], 35),
    "rc64-qtopk10-pcd-ar64": (K10 + BS8 + ["run.prune_score=core_topk", "run.retrieval=frames_cont"] + PCD_AR, 40),
}
runs = [{"name": f"p33-{tag}", "code": CODE, "chunks": 8, "est_min": est, "env": ENV,
         "overrides": BASE + over + [f"paths.save_dir={R}{tag}"]} for tag, (over, est) in CONFIGS.items()]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "priority": 36, "runs": runs}, indent=1))
