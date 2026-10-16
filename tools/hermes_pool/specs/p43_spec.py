"""Phase 43: gold-time upper bounds on S-EMBER with PCD + answer reserve (Qwen3-VL-8B, 475 questions, 64 frames,
timeline prompt, 384 new tokens, repetition penalty 1.1, as the S-EMBER rows).
- oracle-window: all 64 frames inside the gold interval (timestamp text every 5 s outside it), no pruning, no PCD;
  rerun on current code of the phase-21 row (35.3 mIoU).
- oracle-window-pcd-ar64: the same input + PCD (blind prior, alpha 1, adaptive, time scope) + 64-token answer reserve.
- oracle10-pcd-ar64: 64 uniform frames, gold-frame tokens kept first at 10% (oracle selector) + PCD + reserve;
  same budget as the final method (39.3).
The answer reserve lives in the batched decoder (batch size > 1), so the window runs use batch size 3, as the unpruned
final-method rows; oracle-window-bs3 = the no-PCD window at the same batch size (the bs-1 row: 35.3, identical to
phase 21). A first oracle-window-pcd-ar64 attempt at batch size 1 decoded without the reserve and was discarded.
Usage: python3 p43_spec.py > p43.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/"
BASE = ["model=qwen3_vl_8b", "experiment=sember_grounding_uniform", "dataset.max_videos=300", "run.uniform_num_frames=64",
        "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384"]
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time", "run.answer_reserve=64"]
WINDOW = ["run.oracle_window=true", "run.offline_keep_ratio=1.0"]
RUNS = {  # name: (overrides, batch size)
    "oracle-window": (WINDOW, 1),
    "oracle-window-bs3": (WINDOW, 3),
    "oracle-window-pcd-ar64-bs3": (WINDOW + PCD, 3),
    "oracle10-pcd-ar64": (["run.prune_score=oracle", "run.offline_keep_ratio=0.1"] + PCD, 8),
}
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]
runs = [{"name": f"p43-{tag}", "code": CODE, "chunks": 4, "est_min": 40, "priority": 3, "env": ENV,
         "overrides": BASE + over + [f"run.batch_size={bs}", f"paths.save_dir={R}p43-{tag}-timeline-v300"]}
        for tag, (over, bs) in RUNS.items()]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs}, indent=1))
