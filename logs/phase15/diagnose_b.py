"""Phase 15b: literature-inspired diagnostics (Qwen3-VL-8B, S-EMBER grounding, uniform 64 frames).

  1. PERCEPTION vs GENERATION (temporal grounding heads): per-head share of visual attention on gold frames
     while reading the prompt, writing the answer and writing the timestamps (teacher-forced replay).
     Grounding heads are chosen on half of the videos (highest mean lift in the timestamp rows) and
     evaluated on the other half: their lift when the output interval is right (IoU >= 0.5) vs wrong
     (IoU < 0.1) — a gap between what the heads see and what the model writes.
  2. BLIND: answer accuracy and grounding without any video (language prior).
  3. SHUFFLES: frame pairs moved with their timestamps (reading stamps keeps grounding; relying on order
     breaks it), and timestamps alone permuted (does the prediction follow the text or the visuals?).
Writes logs/phase15/diagnose_b.json and prints it. Run from the repo root: python3 logs/phase15/diagnose_b.py
"""
import glob
import hashlib
import json
import random
import statistics as st
import sys

sys.path.insert(0, "logs/phase15")
from diagnose_gpu import metrics, paired, rows_of, run_dir  # noqa: E402


def aattn(tag):
    out = []
    for f in sorted(glob.glob(f"{run_dir(tag)}/aattn-*.jsonl")):
        out += [json.loads(l) for l in open(f)]
    return out


def split(video_id):
    return int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 2


def grounding_heads(tag, k=10):
    recs, rows = aattn(tag), rows_of(tag)
    if not recs or not rows:
        return None
    for r in recs:
        r["_frac"] = len(r["gold_frames"]) / r["num_frames"]
    recs = [r for r in recs if 0 < r["_frac"] < 1 and "time" in r and r["question_id"] in rows]
    layers, heads = len(recs[0]["time"]["gold_share"]), len(recs[0]["time"]["gold_share"][0])
    lift = lambda r, g, l, h: r[g]["gold_share"][l][h] / r["_frac"]
    train = [r for r in recs if split(r["video_id"]) == 0]
    test = [r for r in recs if split(r["video_id"]) == 1]
    score = {(l, h): st.mean(lift(r, "time", l, h) for r in train) for l in range(layers) for h in range(heads)}
    top = sorted(score, key=score.get, reverse=True)[:k]
    head_lift = lambda r, g: st.mean(lift(r, g, l, h) for l, h in top) if g in r else None
    all_lift = lambda r, g: st.mean(lift(r, g, l, h) for l in range(layers) for h in range(heads)) if g in r else None
    good = [r for r in test if rows[r["question_id"]]["_iou"] >= 0.5]
    bad = [r for r in test if rows[r["question_id"]]["_iou"] < 0.1]
    m = lambda xs: round(st.mean(x for x in xs if x is not None), 2) if xs else None
    res = {"questions (test half)": len(test), "grounding heads (layer, head)": [list(x) for x in top],
           "lift, all heads: prompt / answer / time rows": [m([all_lift(r, g) for r in test]) for g in ("prompt", "answer", "time")],
           "lift, grounding heads: prompt / answer / time rows": [m([head_lift(r, g) for r in test]) for g in ("prompt", "answer", "time")],
           "grounding-head lift (prompt rows) when output IoU>=0.5 / IoU<0.1": [m([head_lift(r, "prompt") for r in good]), m([head_lift(r, "prompt") for r in bad])],
           "grounding-head lift (time rows) when output IoU>=0.5 / IoU<0.1": [m([head_lift(r, "time") for r in good]), m([head_lift(r, "time") for r in bad])],
           "wrong outputs (IoU<0.1) whose grounding heads still favour gold (lift>1.5) %":
               round(100 * sum(head_lift(r, "prompt") > 1.5 for r in bad) / max(len(bad), 1), 1),
           "video share of attention: prompt / answer / time rows":
               [round(st.mean(st.mean(r[g]["video_share"]) for r in test if g in r), 3) for g in ("prompt", "answer", "time")]}
    return res, top


def heads_on(tag, top):
    recs, rows = aattn(tag), rows_of(tag)
    if not recs or not rows:
        return None
    vals = {"prompt": [], "time": []}
    for r in recs:
        frac = len(r["gold_frames"]) / r["num_frames"]
        if not 0 < frac < 1 or split(r["video_id"]) != 1:
            continue
        for g in vals:
            if g in r:
                vals[g].append(st.mean(r[g]["gold_share"][l][h] / frac for l, h in top))
    return {"grounding-head lift, prompt / time rows": [round(st.mean(v), 2) if v else None for v in vals.values()]}


def stamp_shuffle(tag):
    """Where do predictions land when only the timestamps are permuted: at the true gold time, or at the
    (wrong) timestamps displayed on the gold frames?"""
    rows = rows_of(tag)
    if not rows:
        return None
    follow_text = follow_visual = n = 0
    for r in rows.values():
        if not r["interval_parseable"]:
            continue
        idx = json.loads(r["source_frame_indices_json"])
        fps = idx[-1] / max(r["_qt"], 1e-6) if idx[-1] else 1.0
        true_t = [i / fps for i in idx]
        pairs = list(range(len(idx) // 2))
        order = pairs[:]
        random.Random(f"{r['video_id']}-{r['question_id']}").shuffle(order)
        shown_idx = [2 * p + k for p in order for k in (0, 1)] + list(range(2 * len(pairs), len(idx)))
        shown_t = [true_t[i] for i in shown_idx]  # timestamp displayed on frame j
        gold = [j for j, t in enumerate(true_t) if r["_gs"] <= t <= r["_ge"]]
        if not gold:
            continue
        ps = min(float(r["pred_start_time"]), float(r["pred_end_time"]))
        pe = max(float(r["pred_start_time"]), float(r["pred_end_time"]))
        mid = (ps + pe) / 2
        text_times = [shown_t[j] for j in gold]
        spacing = 2 * r["_qt"] / len(idx)
        near_text = min(abs(mid - t) for t in text_times) <= spacing
        near_true = r["_gs"] - spacing <= mid <= r["_ge"] + spacing
        follow_text += near_text and not near_true
        follow_visual += near_true and not near_text
        n += 1
    return {"predictions near the displayed (wrong) stamps of gold frames only %": round(100 * follow_text / n, 1),
            "predictions near the true gold time only %": round(100 * follow_visual / n, 1)}


def main():
    out = {}
    res = grounding_heads("dx-aattn-timeline")
    if res:
        out["1_grounding_heads (timeline, unpruned)"], top = res
        off = grounding_heads("dx-aattn")
        out["1b_grounding_heads (official, unpruned)"] = off[0] if off else None
        for name, tag in (("HERMES 10%", "dx-aattn-hermes-keep0.1-timeline"),
                          ("stratified 10%", "dx-aattn-stratified-keep0.1-timeline")):
            out[f"1c_same heads under pruning: {name}"] = heads_on(tag, top)
    base = {"official": rows_of(""), "timeline": rows_of("timeline")}
    out["2_blind"] = {p: metrics(rows_of(f"dx-blind{s}")) for p, s in (("official", ""), ("timeline", "-timeline"))
                      if rows_of(f"dx-blind{s}")}
    out["2b_blind vs video (paired)"] = {p: {k: paired(base[p], rows_of(f"dx-blind{s}"), k) for k in ("acc", "miou", "gq05")}
                                         for p, s in (("official", ""), ("timeline", "-timeline")) if rows_of(f"dx-blind{s}")}
    shuf = {}
    for p, s in (("official", ""), ("timeline", "-timeline")):
        rows = rows_of(f"dx-shufframes{s}")
        if rows:
            shuf[f"frames+stamps shuffled ({p})"] = {"metrics": metrics(rows),
                                                     "video - shuffled (paired)": {k: paired(base[p], rows, k) for k in ("acc", "miou", "gq05")}}
    rows = rows_of("dx-shufstamps")
    if rows:
        shuf["stamps only shuffled (official)"] = {"metrics": metrics(rows), **stamp_shuffle("dx-shufstamps")}
    out["3_shuffles"] = shuf
    json.dump(out, open("logs/phase15/diagnose_b.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
