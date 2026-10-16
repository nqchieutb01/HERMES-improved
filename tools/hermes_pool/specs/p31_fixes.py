"""Phase 31: answer reserve (PCD Time lines cut at the token cap) and a coverage gate for retrieval frames.

Code: worktree branch worktree-p31-gate-pcdfix (commit 0c2f652). Each run differs from an existing run by one
switch: ar64 = run.answer_reserve=64; g = run.retrieval_min_gap=1.0 (retrieval frames only when uniform frames would
be >= 1 s apart: 36/475 S-EMBER questions and all Charades queries fall back to uniform frames).
S-EMBER references: p25-rc64-qtopk10(-pcd), random10-cd-blind-adapt1.0-scopetime. Charades references: phase 28.
Usage: python3 p31_fixes.py > p31_fixes.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
S = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
C = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/charades_sta/"
CD = "/nfs-stor/chieu.nguyen/charades-sta"
COMMON = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", "run.uniform_num_frames=64",
          "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
          "run.batch_size=8", "run.offline_keep_ratio=0.1"]
SEMBER = ["dataset.max_videos=300", "run.retention_snapshot=true"]
CHARADES = [f"dataset.annotation={CD}/charades_sta_test_500.jsonl", f"dataset.video_root={CD}/videos",
            "dataset.max_videos=1000", "dataset.question_categories=[moment_retrieval]"]
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time"]
AR, GATE = ["run.answer_reserve=64"], ["run.retrieval_min_gap=1.0"]
RANDOM = ["run.prune_score=random", "run.prefetch_videos=4"]
SEMVID = ["run.prune_score=semvid", "run.prefetch_videos=4"]
RC = ["run.prune_score=core_topk", "run.retrieval=frames_cont"]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
CENV = {**ENV, "HERMES_RETRIEVAL_INDEX": f"{CD}/siglip2_1fps"}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]

runs = []
for tag, over in {"rc64-qtopk10-pcd-ar64": RC + PCD + AR, "random10-pcd-ar64": RANDOM + PCD + AR,
                  "rc64g-qtopk10": RC + GATE, "rc64g-qtopk10-pcd-ar64": RC + GATE + PCD + AR}.items():
    runs.append({"name": f"p31-{tag}", "code": CODE, "chunks": 4, "est_min": 50, "env": ENV,
                 "overrides": COMMON + SEMBER + over + [f"paths.save_dir={S}p31-{tag}-timeline-v300-native-time"]})
for tag, over in {"random10-pcd-ar64": RANDOM + PCD + AR, "rc64-qtopk10-pcd-ar64": RC + PCD + AR,
                  "semvid10-pcd-ar64": SEMVID + PCD + AR, "rc64g-qtopk10": RC + GATE,
                  "rc64g-qtopk10-pcd-ar64": RC + GATE + PCD + AR}.items():
    runs.append({"name": f"p31c-{tag}", "code": CODE, "chunks": 4, "est_min": 25, "env": CENV,
                 "overrides": COMMON + CHARADES + over + [f"paths.save_dir={C}p31-{tag}"]})

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "priority": 35, "runs": runs}, indent=1))
