"""Find QVHighlights questions where CoCo grounds well and the baselines do not (teaser candidates)."""
import csv
import sys

sys.path.insert(0, "/home/chieu.nguyen/HERMES/.claude/worktrees/paper-draft/docs/paper/analysis")
from floors import R, load, iou  # noqa: E402

B = R + "qvhighlights/"
NAMES = ["random10", "unpruned", "semvid10-pcd-ar64", "rc64-qtopk10", "random10-pcd-ar64", "rc64-qtopk10-pcd-ar64"]
runs = {k: load(B + k) for k in NAMES}
text = {r["question_id"]: (r["question"], r["video_id"], r["pred_raw"]) for r in csv.DictReader(open(B + "random10/results.csv"))}
raw = {n: {r["question_id"]: r["pred_raw"] for r in csv.DictReader(open(B + n + "/results.csv"))} for n in NAMES}
cands = []
for k, g in runs["random10"].items():
    i = {n: iou(d[k][0], d[k][1], d[k][2]) for n, d in runs.items()}
    if (i["rc64-qtopk10-pcd-ar64"] > 0.75 and i["random10"] < 0.15 and i["unpruned"] < 0.2
            and i["semvid10-pcd-ar64"] < 0.3 and g[2] - g[1] < 0.4 * g[3] and g[1] > 0.2 * g[3]):
        cands.append((i["rc64-qtopk10-pcd-ar64"], k))
for s, k in sorted(cands, reverse=True)[:6]:
    g = runs["random10"][k]
    print(f"\n{s:.2f} {k} video={text[k][1]} | {text[k][0][:120]}")
    print(f"   gold [{g[1]}, {g[2]}] window {g[3]}")
    for n, d in runs.items():
        print(f"   {n:24s} {d[k][0]}  IoU {iou(d[k][0], d[k][1], d[k][2]):.2f}")
