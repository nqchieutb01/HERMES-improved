"""Phase 40 runs in our pipeline (TimeLens prompt and protocol unless stated).

check: validation runs that gate the sweeps —
  bridge-<ds>: Qwen3-VL-8B unpruned under the per-video TimeLens protocol (protocol_fps=2, 14,336 tokens, min 64 per
  frame) on a fixed 400-query slice of Charades- / ActivityNet-TimeLens (lowest question ids), compared paired with
  TimeLens' own script on the same queries (full-set anchor runs);
  sember-t8-unpruned: TimeLens-8B on S-EMBER with exactly the settings of the Qwen3-VL-8B unpruned run
  (64 frames, timeline prompt, 384 tokens, repetition penalty 1.1).
sember: TimeLens-8B on S-EMBER at 10% with the configurations of the Qwen3-VL-8B rows (random 10%, Retrieve + Cover
  10%, + PCD + answer reserve, random + PCD + reserve), same pre-resize as the unpruned check.
sweep: full TimeLens-Bench (Charades 3,363 / ActivityNet 4,500 / QVHighlights 1,541 queries), both models, protocol
  input, TimeLens prompt: Cover 31% and random 31%. Matching TimeLens-script baselines (2 fps, min 16 tokens, total
  3,500 / 4,800 / 4,500 -> 3,157 / 3,981 / 4,197 mean visual tokens vs 31% of the protocol 3,133 / 4,012 / 4,229;
  metadata count on 120 videos per dataset) are tasks in tl_full_tasks.txt.
Usage: python3 p40_spec.py check|sember|sweep > p40_<set>.json
"""
import json
import sys

CODE = "/home/chieu.nguyen/HERMES/.claude/worktrees/p31-gate-pcdfix"
D = "/nfs-stor/chieu.nguyen/timelens-bench"
RES = "/home/chieu.nguyen/HERMES/results/"
T8 = "/nfs-stor/chieu.nguyen/models/TimeLens-8B"
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
TL = ["experiment=sember_grounding_uniform", "dataset.max_videos=100000", "dataset.question_categories=[moment_retrieval]",
      "dataset.grounding_prompt=timelens", "run.keep_time_tokens=surviving", "run.max_new_tokens=512",
      "run.repetition_penalty=1.0", "run.uniform_num_frames=64", "run.protocol_fps=2", "run.protocol_total_tokens=14336",
      "run.protocol_min_tokens=64", "run.prefetch_videos=4"]
SEMBER_UNPRUNED = ["dataset.max_videos=300", "experiment=sember_grounding_uniform", "run.uniform_num_frames=64",
                   "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
                   "run.offline_keep_ratio=1.0"]
SLOTS = json.load(open("/home/chieu.nguyen/hermes_pool/slots.json"))["slots"]


def tl_run(name, model, ds, jsonl, over, bs, chunks, est, prio):
    mt = "qwen3_vl_8b" if model is None else "timelens_8b"
    return {"name": name, "code": CODE, "chunks": chunks, "est_min": est, "priority": prio, "env": ENV,
            "overrides": ["model=qwen3_vl_8b"] + ([f"model.model_path={model}"] if model else []) + TL
            + [f"dataset.annotation={D}/{jsonl}", f"dataset.video_root={D}/videos/{ds}/{ds}"] + over
            + [f"run.batch_size={bs}", f"paths.save_dir={RES}{mt}/timelens_{ds}/{name.split('-', 1)[1]}"]}


CHECK = [
    tl_run("p40-bridge-charades", None, "charades", "charades_bridge400.jsonl",
           ["run.prune_score=random", "run.offline_keep_ratio=1.0"], 1, 4, 30, 7),
    tl_run("p40-bridge-activitynet", None, "activitynet", "activitynet_bridge400.jsonl",
           ["run.prune_score=random", "run.offline_keep_ratio=1.0"], 1, 8, 40, 7),
    {"name": "p40-sember-t8-unpruned", "code": CODE, "chunks": 4, "est_min": 60, "priority": 8, "env": ENV,
     "overrides": ["model=qwen3_vl_8b", f"model.model_path={T8}"] + SEMBER_UNPRUNED
     # TimeLens-8B's saved processor computes a resized grid but does not resize high-resolution frames (shape error);
     # pre-resize to the size Qwen3-VL's processor gives a 16-frame chunk (longest_edge 25,165,824 / 16), same grid.
     + ["run.frame_max_pixels=1572864", "run.frame_min_pixels=0"]
     + [f"paths.save_dir={RES}timelens_8b/sember_grounding/unpruned-timeline-v300"]},
]
T8_FRAMES = ["run.frame_max_pixels=1572864", "run.frame_min_pixels=0"]
SEMBER10 = ["dataset.max_videos=300", "experiment=sember_grounding_uniform", "run.uniform_num_frames=64",
            "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
            "run.offline_keep_ratio=0.1", "run.batch_size=8", "run.retention_snapshot=true"]
RC = ["run.prune_score=core_topk", "run.retrieval=frames_cont"]
PCD = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
       "run.contrastive_scope=time", "run.answer_reserve=64"]
SEMBER = [{"name": f"p40-sember-t8-{n}", "code": CODE, "chunks": 4, "est_min": 50, "priority": 9, "env": ENV,
           "overrides": ["model=qwen3_vl_8b", f"model.model_path={T8}"] + SEMBER10 + T8_FRAMES + o
           + [f"paths.save_dir={RES}timelens_8b/sember_grounding/{n}-timeline-v300"]}
          for n, o in [("random10", ["run.prune_score=random", "run.prefetch_videos=4"]), ("rc64-qtopk10", RC),
                       ("rc64-qtopk10-pcd-ar64", RC + PCD),
                       ("random10-pcd-ar64", ["run.prune_score=random", "run.prefetch_videos=4"] + PCD)]]
FULL = {"charades": (12, 8, 4, 0), "qvhighlights": (6, 8, 4, 0), "activitynet": (18, 4, 1, 2)}  # chunks, bs, prefetch, prio+
SWEEP = [tl_run(f"p40-full{'-t8' if model else ''}-{ds}-{sel}31", model, ds, f"{ds}_full.jsonl",
                [f"run.prune_score={score}", "run.offline_keep_ratio=0.31", f"run.prefetch_videos={pf}"], bs, ch, 45,
                16 + (1 if model else 0) + extra)
         for model in (None, T8) for ds, (ch, bs, pf, extra) in FULL.items()
         for sel, score in (("cover", "core_topk"), ("random", "random"))]
for r in SWEEP:  # the per-dataset prefetch replaces the one in TL
    r["overrides"].remove("run.prefetch_videos=4")

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": {"check": CHECK, "sember": SEMBER, "sweep": SWEEP}[sys.argv[1]]}, indent=1))
