"""Score the QVHighlights one-line runs that bear on Table 3 (main table), 2026-10-17 afternoon.
Reuses score_dir/anchor/logs/mets/boot from collect4/score_tl.py unchanged (TimeLens convert() semantics)."""
import json
import os
import sys

import numpy as np

src = open("/home/chieu.nguyen/.claude/jobs/7e393a16/tmp/collect4/score_tl.py").read()
src = src[:src.index("res = {\"cells\"")]          # functions and constants only, no module-level scoring
exec(compile(src, "score_tl_funcs", "exec"))

LOC = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/timelens_qvhighlights/"
HN = "/home/chieu.nguyen/.claude/jobs/7e393a16/tmp/t3/hn/qvh/"
RUNS = {
    "anchor (TL script)": ("tl", "/home/chieu.nguyen/hermes_pool/timelens/runs_full/qwen3/qvhighlights/anchor/all.jsonl"),
    "cover31": ("ours", LOC + "full-qvhighlights-cover31"),
    "random31": ("ours", LOC + "full-qvhighlights-random31"),
    "rc64-cover23": ("ours", HN + "full-rc64-cover23"),
    "rc64-random23": ("ours", HN + "full-rc64-random23"),
    "rc64-qtopk23": ("ours", HN + "full-rc64-qtopk23"),
    "rc64-snapkv23": ("ours", HN + "full-rc64-snapkv23"),
    "rc64-fastvid19": ("ours", LOC + "full-rc64-fastvid19"),
    "rc64-semvid19": ("ours", LOC + "full-rc64-semvid19"),
    "u64-cover10-fit": ("ours", HN + "full-u64-cover10-fit"),
    "u64-random10-fit": ("ours", HN + "full-u64-random10-fit"),
    "u64-cover5-fit": ("ours", HN + "full-u64-cover5-fit"),
    "u64-random5-fit": ("ours", HN + "full-u64-random5-fit"),
}
out, per = {}, {}
for name, (kind, p) in RUNS.items():
    if kind == "tl":
        d, info = anchor(p)
    else:
        d, info = score_dir(p, "qvhighlights")
        info.update(logs(p))
    per[name] = d
    info["n"] = len(d)
    if d:
        info.update(mets(list(d.values())))
    out[name] = info
    cfg = (info.get("configs") or [{}])[0]
    print(f"{name:20s} n={len(d):5d}  mIoU={info.get('mIoU')}  R@0.5={info.get('R@0.5')}  bs={info.get('batch_size')}  "
          f"kept={info.get('kept_mean')}/{info.get('total_mean')}  sel={cfg.get('prune_score')} keep={cfg.get('offline_keep_ratio')} "
          f"ret={cfg.get('retrieval')} frames={cfg.get('uniform_num_frames')} prompt={cfg.get('grounding_prompt')} "
          f"ktt={cfg.get('keep_time_tokens')} unparsed={info.get('unparsed')}")

print()
for a, b in [("rc64-cover23", "rc64-random23"), ("rc64-cover23", "rc64-fastvid19"), ("rc64-cover23", "rc64-semvid19"),
             ("rc64-cover23", "rc64-qtopk23"), ("rc64-cover23", "rc64-snapkv23"), ("u64-cover10-fit", "u64-random10-fit"),
             ("u64-cover10-fit", "anchor (TL script)"), ("u64-cover5-fit", "u64-random5-fit")]:
    A, B = per[a], per[b]
    q = sorted(set(A) & set(B))
    if not q:
        continue
    x = np.array([A[k] for k in q]); y = np.array([B[k] for k in q])
    print(f"{a} - {b}: n={len(q)} mIoU {boot(x - y)}  R@0.5 {boot((x >= .5).astype(float) - (y >= .5).astype(float))}")
json.dump(out, open("/home/chieu.nguyen/.claude/jobs/7e393a16/tmp/t3/t3.json", "w"), indent=1, default=str)
