"""Phase 36: coverage vs per-frame evidence at a fixed token budget (fewer frames, more tokens per frame).

Every run keeps the same absolute number of visual tokens as a 64-frame run it is compared with: F frames at keep
ratio r * 64 / F, so only the split between frames and tokens per frame changes.
- Charades-STA (30 s videos): 32 frames at 20% = the budget of 64 frames at 10% (phase 28/31 lineup). Uniform frames
  (retrieval is off on these videos: 32 frames over 30 s are < 1 s apart).
- QVHighlights at the budget where the 64-frame final method collapsed (64 frames at 5%, ~260 tokens):
  32 frames at 10%, 16 frames at 20%.
- ActivityNet at the budget where it collapsed (64 frames at 10%, ~120-300 tokens): 32 frames at 20%, 16 at 40%.
Code: worktree branch worktree-p31-gate-pcdfix (no new options). Usage: python3 p36_frames.py > p36_frames.json
"""
import importlib.util
import json


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = "/home/chieu.nguyen/hermes_pool/specs/"
p31, p33, p34 = load("p31", P + "p31_fixes.py"), load("p33", P + "p33_qvh.py"), load("p34", P + "p34_anet.py")
PCD_AR = p31.PCD + p31.AR
RANDOM, SEMVID = ["run.prune_score=random", "run.prefetch_videos=4"], ["run.prune_score=semvid", "run.prefetch_videos=4"]
COVER_U = ["run.prune_score=core_topk", "run.prefetch_videos=4"]
COVER_R = ["run.prune_score=core_topk", "run.retrieval=frames_cont"]


def frames(over, n, keep):
    """Replace frame count and keep ratio in a base override list."""
    out = [o for o in over if not o.startswith(("run.uniform_num_frames=", "run.offline_keep_ratio=", "run.batch_size="))]
    return out + [f"run.uniform_num_frames={n}", f"run.offline_keep_ratio={keep}"]


runs = []
# Charades: 32 frames at 20% (= 64 frames at 10%).
C_BASE = p31.COMMON + p31.CHARADES
for tag, over, bs, est in [("unpruned", RANDOM, 1, 30), ("random", RANDOM, 8, 15), ("random-pcd-ar64", RANDOM + PCD_AR, 8, 20),
                           ("semvid", SEMVID, 8, 15), ("semvid-pcd-ar64", SEMVID + PCD_AR, 8, 20),
                           ("u32-qtopk", COVER_U, 8, 15), ("u32-qtopk-pcd-ar64", COVER_U + PCD_AR, 8, 20)]:
    keep = 1.0 if tag == "unpruned" else 0.2
    base, pcd = (tag[:-len("-pcd-ar64")], "-pcd-ar64") if tag.endswith("-pcd-ar64") else (tag, "")
    name = f"f32-{base}" + ("" if tag == "unpruned" else "20") + pcd
    runs.append({"name": f"p36c-{name}", "code": p31.CODE, "chunks": 4, "est_min": est, "priority": 25, "env": p31.CENV,
                 "overrides": frames(C_BASE + over, 32, keep) + [f"run.batch_size={bs}", f"paths.save_dir={p31.C}{name}"]})
# QVHighlights (budget of 64 frames at 5%) and ActivityNet (budget of 64 frames at 10%).
for ds, mod, base_keep, chunks in (("q", p33, 0.05, 8), ("a", p34, 0.10, 8)):
    for n in (32, 16):
        keep = round(base_keep * 64 / n, 3)
        for tag, over in [("random", RANDOM + mod.PCD_AR), ("semvid", SEMVID + mod.PCD_AR),
                          (f"rc{n}-qtopk", COVER_R + mod.PCD_AR)]:
            name = f"f{n}-{tag}{int(round(keep * 100))}-pcd-ar64"
            runs.append({"name": f"p36{ds}-{name}", "code": mod.CODE, "chunks": chunks, "est_min": 25, "priority": 26,
                         "env": mod.ENV, "overrides": frames(mod.BASE + over, n, keep)
                         + ["run.batch_size=8", f"paths.save_dir={mod.R}{name}"]})

if __name__ == "__main__":
    print(json.dumps({"slots": p31.SLOTS, "runs": runs}, indent=1))
