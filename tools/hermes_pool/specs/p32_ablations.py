"""Phase 32 (S-EMBER, 300 videos, 475 questions, 64 frames): ablations for the final method.

shared: the same kept tokens in every layer (run.shared_selection; one random draw / layer-averaged attention),
    against the per-layer selection of random10 and p25-rc64-qtopk10 (and their PCD + answer-reserve versions, p31).
budget: the final decoding (PCD + answer reserve 64) at 5% and 20% for random and Retrieve + Cover; p30 ran these
    budgets without the reserve, p31 at 10% with it.
seeds: random10 with two more selection seeds (run.prune_seed), for the variance of the random baseline.
Code: worktree branch worktree-p31-gate-pcdfix. Usage: python3 p32_ablations.py > p32_ablations.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
S = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform", "run.uniform_num_frames=64",
        "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
        "run.batch_size=8", "run.retention_snapshot=true"]
PCD_AR = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
          "run.contrastive_scope=time", "run.answer_reserve=64"]
RANDOM = ["run.prune_score=random", "run.prefetch_videos=4"]
RC = ["run.prune_score=core_topk", "run.retrieval=frames_cont"]
SHARED = ["run.shared_selection=true"]
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]

RUNS = {  # name: (overrides, priority)
    "random10-shared": (RANDOM + SHARED + ["run.offline_keep_ratio=0.1"], 37),
    "random10-shared-pcd-ar64": (RANDOM + SHARED + PCD_AR + ["run.offline_keep_ratio=0.1"], 37),
    "rc64-qtopk10-shared": (RC + SHARED + ["run.offline_keep_ratio=0.1"], 37),
    "rc64-qtopk10-shared-pcd-ar64": (RC + SHARED + PCD_AR + ["run.offline_keep_ratio=0.1"], 37),
    "rc64-qtopk5-pcd-ar64": (RC + PCD_AR + ["run.offline_keep_ratio=0.05"], 38),
    "rc64-qtopk20-pcd-ar64": (RC + PCD_AR + ["run.offline_keep_ratio=0.2"], 38),
    "random5-pcd-ar64": (RANDOM + PCD_AR + ["run.offline_keep_ratio=0.05"], 38),
    "random20-pcd-ar64": (RANDOM + PCD_AR + ["run.offline_keep_ratio=0.2"], 38),
    "random10-seed1": (RANDOM + ["run.prune_seed=1", "run.offline_keep_ratio=0.1"], 45),
    "random10-seed2": (RANDOM + ["run.prune_seed=2", "run.offline_keep_ratio=0.1"], 45),
}
runs = [{"name": f"p32-{tag}", "code": CODE, "chunks": 4, "est_min": 50, "priority": prio,
         "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"},
         "overrides": BASE + over + [f"paths.save_dir={S}p32-{tag}-timeline-v300-native-time"]}
        for tag, (over, prio) in RUNS.items()]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs}, indent=1))
