"""Video-free floors on all four datasets (CPU, saved outputs). m:ss-aware parsing (same for every run)."""
import csv, re, sys, random
sys.path.insert(0, "/home/chieu.nguyen/HERMES/eval/sember")
from eval_grounding import parse_grounding_response as official, temporal_iou


def parse(t):
    o = official(t)
    if o[1] is None:
        t = re.sub(r"\b(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\b", lambda m: f"{3600*int(m[1])+60*int(m[2])+float(m[3]):.1f}", t)
        o = official(re.sub(r"\b(\d{1,2}):(\d{1,2}(?:\.\d+)?)\b", lambda m: f"{60*int(m[1])+float(m[2]):.1f}", t))
    return o[1], o[2]


R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/"
S = R + "sember_grounding/uniform-n64-time-count-location-"
SETS = {
    "S-EMBER": {"Random": S + "offline-random-keep0.1-timeline-bs8-v300-native-time",
                "SemVID+PCD+res": S + "p35-u64-semvid10-pcd-ar64-timeline-v300-native-time",
                "Random+PCD+res": S + "p31-random10-pcd-ar64-timeline-v300-native-time",
                "RC (no PCD)": S + "p25-rc64-qtopk10-timeline-v300-native-time",
                "final": S + "p31-rc64-qtopk10-pcd-ar64-timeline-v300-native-time",
                "Unpruned": S + "timeline-v300-native-time"},
    "QVH": {k: R + "qvhighlights/" + v for k, v in {"Random": "random10", "SemVID+PCD+res": "semvid10-pcd-ar64",
            "Random+PCD+res": "random10-pcd-ar64", "RC (no PCD)": "rc64-qtopk10", "final": "rc64-qtopk10-pcd-ar64",
            "Unpruned": "unpruned"}.items()},
    "ANet30": {k: R + "activitynet/" + v for k, v in {"Random": "random30", "SemVID+PCD+res": "semvid30-pcd-ar64",
               "Random+PCD+res": "random30-pcd-ar64", "RC (no PCD)": "rc64-qtopk30", "final": "rc64-qtopk30-pcd-ar64",
               "Unpruned": "unpruned"}.items()},
    "Charades": {k: R + "charades_sta/" + v for k, v in {"Random": "random10", "SemVID+PCD+res": "p31-semvid10-pcd-ar64",
                 "Random+PCD+res": "p31-random10-pcd-ar64", "RC (no PCD)": "rc64-qtopk10",
                 "final(gated)": "p31-rc64g-qtopk10-pcd-ar64", "Unpruned": "unpruned-bs1"}.items()},
}


def load(p):
    rows = {}
    for r in csv.DictReader(open(p + "/results.csv")):
        q = float(r.get("question_time") or r["duration"])
        rows[r["question_id"]] = (parse(r["pred_raw"]), float(r["answer_start_time"]), float(r["answer_end_time"]), q,
                                  r.get("question_category", ""))
    return rows


def iou(p, s, e):
    return temporal_iou(p[0], p[1], s, e) if p[0] is not None else 0.0


def mean(v):
    return 100 * sum(v) / len(v) if v else float("nan")


for ds, runs in SETS.items():
    data = {k: load(v) for k, v in runs.items()}
    ref = data["Random"]
    qids = sorted(ref)
    closed = [k for k in qids if ref[k][2] < 0.8 * ref[k][3]]
    print(f"\n## {ds}: n={len(qids)}, closed (gold end < 0.8q) = {len(closed)}, "
          f"mean gold len / q = {sum((ref[k][2]-ref[k][1])/ref[k][3] for k in qids)/len(qids):.2f}, "
          f"gold end >= 0.8q: {100*(1-len(closed)/len(qids)):.0f}%")
    best = max(((a, mean([temporal_iou(a * ref[k][3], ref[k][3], ref[k][1], ref[k][2]) for k in qids])) for a in
                [i / 20 for i in range(20)]), key=lambda x: x[1])
    # best constant interval of any position/length (oracle over a 20x20 grid of fractions of q)
    grid = max(((a, b, mean([temporal_iou(a * ref[k][3], b * ref[k][3], ref[k][1], ref[k][2]) for k in qids]))
                for a in [i / 20 for i in range(20)] for b in [j / 20 for j in range(1, 21)] if b > a), key=lambda x: x[2])
    gridc = max(((a, b, mean([temporal_iou(a * ref[k][3], b * ref[k][3], ref[k][1], ref[k][2]) for k in closed]))
                 for a in [i / 20 for i in range(20)] for b in [j / 20 for j in range(1, 21)] if b > a), key=lambda x: x[2])
    print(f"best constant [a q, q]: a={best[0]:.2f} mIoU {best[1]:.1f};  best constant [a q, b q] (oracle grid): "
          f"[{grid[0]:.2f},{grid[1]:.2f}] {grid[2]:.1f};  on closed: [{gridc[0]:.2f},{gridc[1]:.2f}] {gridc[2]:.1f}")
    print("| run | mIoU all | snap-end mIoU | mIoU closed | snap closed |")
    for name, d in data.items():
        a = [iou(d[k][0], d[k][1], d[k][2]) for k in qids]
        sn = [iou((d[k][0][0], d[k][3]), d[k][1], d[k][2]) if d[k][0][0] is not None else 0.0 for k in qids]
        c = [iou(d[k][0], d[k][1], d[k][2]) for k in closed]
        snc = [iou((d[k][0][0], d[k][3]), d[k][1], d[k][2]) if d[k][0][0] is not None else 0.0 for k in closed]
        print(f"| {name} | {mean(a):.1f} | {mean(sn):.1f} | {mean(c):.1f} | {mean(snc):.1f} |")
