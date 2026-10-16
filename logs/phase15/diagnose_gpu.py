"""Phase 15: analyse the diagnostic GPU runs (Qwen3-VL-8B, S-EMBER grounding, uniform 64 frames).

  1. QUESTION ATTENTION: does the real question attend to the gold frames? Attention share on gold frames vs
     their share of frames (lift), per layer band; top-8 frame hit rate; attention by frame position
     (first-frame sink); link to answer correctness.
  2. RETENTION: what pruning keeps. Tokens kept on gold vs other frames, gold frames left with no token,
     gold timestamps kept; per pruning method; link to IoU.
  3. SELECTION BOTTLENECK: oracle pruning (gold frames first) vs HERMES / stratified / random at the same
     budget and vs unpruned, paired per question.
  4. SCORER: hermes_exact (probe questions propagated through every layer) vs the default HERMES scorer.
  5. TIMESTAMP USE: +200 s shifted timestamps (IoU after undoing the shift, share of shifted predictions)
     and no timestamp text.
  6. PRIMACY: frames start at half the question time; where predictions start relative to the shown window,
     and answers for evidence that lies entirely before the window.
Writes logs/phase15/diagnose_gpu.json and prints it. Run from the repo root: python3 logs/phase15/diagnose_gpu.py
"""
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, "logs/phase13")
from compare import boot  # noqa: E402

B = "results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
T = "-v300-native-time"


def run_dir(tag):
    return f"{B}{tag}{T}" if tag else f"{B}{T[1:]}"


def rows_of(tag):
    path = f"{run_dir(tag)}/sember_grounding_scored.jsonl"
    if not os.path.exists(path):
        return None
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(path)}
    jpath = f"{run_dir(tag)}/answer_judgments.jsonl"
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(jpath)} if os.path.exists(jpath) else {}
    for q, r in rows.items():
        r["_ok"] = judge.get(q)
        r["_iou"] = float(r["temporal_iou"] or 0)
        r["_gs"], r["_ge"], r["_qt"] = float(r["answer_start_time"]), float(r["answer_end_time"]), float(r["question_time"])
    return rows


def metrics(rows):
    v = list(rows.values())
    has_judge = all(r["_ok"] is not None for r in v)
    return {"n": len(v), "acc": round(100 * sum(r["_ok"] for r in v) / len(v), 1) if has_judge else None,
            "miou": round(100 * sum(r["_iou"] for r in v) / len(v), 1),
            "r05": round(100 * sum(r["_iou"] >= 0.5 for r in v) / len(v), 1),
            "gq05": round(100 * sum(bool(r["_ok"]) and r["_iou"] >= 0.5 for r in v) / len(v), 1) if has_judge else None}


def paired(a, b, key):
    qs = sorted(set(a) & set(b))
    f = {"acc": lambda r: 100.0 * bool(r["_ok"]), "miou": lambda r: 100 * r["_iou"],
         "gq05": lambda r: 100.0 * (bool(r["_ok"]) and r["_iou"] >= 0.5)}[key]
    d = [f(a[q]) - f(b[q]) for q in qs]
    lo, hi = boot(d, n=4000)
    return f"{sum(d) / len(d):+.1f} [{lo:+.1f}, {hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else "")


def jsonl(tag, prefix):
    out = []
    for f in sorted(glob.glob(f"{run_dir(tag)}/{prefix}-*.jsonl")):
        out += [json.loads(l) for l in open(f)]
    return out


def gold_frames(times, gs, ge):
    return [i for i, t in enumerate(times) if gs <= t <= ge]


def question_attention():
    recs = jsonl("dx-qattn", "qattn")
    rows = rows_of("dx-qattn") or {}
    if not recs:
        return None
    bands = ("early", "mid", "late")
    lift = {b: [] for b in bands}
    share_gold = {b: [] for b in bands}
    hit8, first_share, stamp_share, by_ok = [], {b: [] for b in bands}, {b: [] for b in bands}, {True: [], False: []}
    decile = [[0.0] * 10 for _ in bands]
    for rec in recs:
        r = rows.get(rec["question_id"])
        if not r:
            continue
        times = rec["frame_times"]
        vis = {int(k): v for k, v in rec["visual"].items()}
        stamps = {int(k): v for k, v in rec["timestamps"].items()}
        gold = set(gold_frames(times, r["_gs"], r["_ge"]))
        if not gold or len(gold) == len(times):
            continue
        frac = len(gold) / len(times)
        for bi, b in enumerate(bands):
            tot = sum(v[bi] for v in vis.values())
            g = sum(vis[i][bi] for i in gold if i in vis)
            share_gold[b].append(g / tot)
            lift[b].append((g / tot) / frac)
            first_share[b].append(sum(vis[i][bi] for i in (0, 1) if i in vis) / tot)
            stamp_tot = sum(v[bi] for v in stamps.values())
            stamp_share[b].append(stamp_tot / (stamp_tot + tot))
            for i, v in vis.items():
                decile[bi][min(9, i * 10 // len(times))] += v[bi] / tot / len(recs)
        order = sorted(vis, key=lambda i: -vis[i][2])[:8]
        hit8.append(any(i in gold for i in order))
        if r["_ok"] is not None:
            by_ok[r["_ok"]].append(lift["late"][-1])
    if not hit8:
        return None  # run not finished yet
    m = lambda xs: round(st.mean(xs), 3) if xs else None
    return {
        "questions": len(hit8),
        "attention share on gold frames (early/mid/late)": [m(share_gold[b]) for b in bands],
        "lift = share on gold / gold share of frames (1 = no targeting)": [m(lift[b]) for b in bands],
        "median lift (early/mid/late)": [round(st.median(lift[b]), 3) for b in bands],
        "top-8 late-attention frames include a gold frame %": round(100 * sum(hit8) / len(hit8), 1),
        "attention on the first frame pair (frames 0-1, 3% of frames)": [m(first_share[b]) for b in bands],
        "timestamp-text share of video attention": [m(stamp_share[b]) for b in bands],
        "visual attention by frame decile (late layers)": [round(x, 3) for x in decile[2]],
        "late lift when answer correct / wrong": [m(by_ok[True]), m(by_ok[False])],
    }


def retention(tag):
    recs = jsonl(tag, "retention")
    rows = rows_of(tag) or {}
    stats = []
    for rec in recs:
        r = rows.get(rec["question_id"])
        if not r:
            continue
        times = rec["frame_times"]
        enc = {int(k): v for k, v in rec["encoded"].items()}
        kept = {int(k): v for k, v in rec["kept"].items()}
        tk = {int(k): v for k, v in rec["time_kept"].items()}
        gold = gold_frames(times, r["_gs"], r["_ge"])
        other = [i for i in range(len(times)) if i not in set(gold)]
        if not gold or not other:
            continue
        share = lambda ids: sum(kept.get(i, 0) for i in ids) / max(sum(enc.get(i, 0) for i in ids), 1)
        dead = lambda ids: sum(kept.get(i, 0) < 1 for i in ids) / len(ids)
        gold_groups = {i - i % 2 for i in gold}
        stats.append({"g": share(gold), "o": share(other), "gd": dead(gold), "od": dead(other),
                      "gstamp": sum(tk.get(g, 0) > 0 for g in gold_groups) / len(gold_groups),
                      "iou": r["_iou"], "ok": r["_ok"]})
    if not stats:
        return None
    m = lambda xs, k: round(100 * st.mean(x[k] for x in xs), 1) if xs else None
    hi = [x for x in stats if x["iou"] >= 0.3]
    lo = [x for x in stats if x["iou"] < 0.3]
    return {"questions": len(stats), "gold frames: % tokens kept": m(stats, "g"), "other frames: % tokens kept": m(stats, "o"),
            "gold frames with no token %": m(stats, "gd"), "other frames with no token %": m(stats, "od"),
            "gold groups keeping their timestamp %": m(stats, "gstamp"),
            "gold % kept when IoU>=0.3 / <0.3": [m(hi, "g"), m(lo, "g")],
            "gold frames with no token when IoU>=0.3 / <0.3": [m(hi, "gd"), m(lo, "gd")]}


def shifted(tag, offset):
    rows = rows_of(tag)
    if not rows:
        return None
    moved = fixed = 0
    ious = []
    for r in rows.values():
        ps, pe = r.get("pred_start_time"), r.get("pred_end_time")
        if not r["interval_parseable"] or ps in (None, ""):
            ious.append(0.0)
            continue
        ps, pe = sorted((float(ps), float(pe)))
        moved += ps >= offset * 0.75
        s, e = ps - offset, pe - offset
        inter = max(0.0, min(e, r["_ge"]) - max(s, r["_gs"]))
        union = max(e, r["_ge"]) - min(s, r["_gs"])
        ious.append(inter / union if union > 0 else 0.0)
    n = len(rows)
    return {"predictions starting >= 0.75 x offset (model followed the shifted stamps) %": round(100 * moved / n, 1),
            "mIoU after undoing the shift": round(100 * sum(ious) / n, 1),
            "R@0.5 after undoing the shift": round(100 * sum(i >= 0.5 for i in ious) / n, 1)}


def primacy(tag):
    rows = rows_of(tag)
    if not rows:
        return None
    starts, before, answered_before = [], 0, []
    for r in rows.values():
        w0 = 0.5 * r["_qt"]
        if r["interval_parseable"] and r.get("pred_start_time") not in (None, ""):
            ps = min(float(r["pred_start_time"]), float(r["pred_end_time"]))
            starts.append((ps - w0) / max(r["_qt"] - w0, 1))
        if r["_ge"] < w0:
            before += 1
            if r["_ok"] is not None:
                answered_before.append(r["_ok"])
    return {"median predicted start, position in shown window (0 = window start)": round(st.median(starts), 2),
            "predictions starting in the first 10% of the window %": round(100 * sum(s <= 0.1 for s in starts) / len(starts), 1),
            "predictions starting before the window (impossible) %": round(100 * sum(s < -0.02 for s in starts) / len(starts), 1),
            "questions whose gold interval ends before the window %": round(100 * before / len(rows), 1),
            "answer accuracy on those (evidence not shown)": round(100 * st.mean(answered_before), 1) if answered_before else None}


def main():
    out = {"1_question_attention": question_attention()}
    out["2_retention"] = {name: retention(tag) for name, tag in (
        ("HERMES 10%", "dx-ret-hermes-keep0.1"), ("HERMES 5%", "dx-ret-hermes-keep0.05"),
        ("stratified 10%", "dx-ret-stratified-keep0.1"), ("random 10%", "dx-ret-random-keep0.1"),
        ("oracle 10%", "offline-oracle-keep0.1"), ("oracle 5%", "offline-oracle-keep0.05"),
        ("hermes_exact 10%", "offline-hermes_exact-keep0.1"), ("hermes_exact 5%", "offline-hermes_exact-keep0.05"))}
    runs = {"unpruned": "", "HERMES 10%": "offline-hermes-keep0.1-", "HERMES 5%": "offline-hermes-keep0.05-",
            "stratified 10%": "offline-stratified-keep0.1-", "stratified 5%": "offline-stratified-keep0.05-",
            "random 10%": "offline-random-keep0.1-", "random 5%": "offline-random-keep0.05-",
            "oracle 10%": "offline-oracle-keep0.1-", "oracle 5%": "offline-oracle-keep0.05-",
            "hermes_exact 10%": "offline-hermes_exact-keep0.1-", "hermes_exact 5%": "offline-hermes_exact-keep0.05-"}
    table = {}
    loaded = {}
    for prompt, suffix in (("official", ""), ("timeline", "timeline-")):
        for name, pre in runs.items():
            rows = rows_of((pre + suffix).rstrip("-"))
            if rows:
                loaded[(prompt, name)] = rows
                table[f"{name} ({prompt})"] = metrics(rows)
    out["3_selection_and_scorer_metrics"] = table
    comps = {}
    for prompt in ("official", "timeline"):
        for a, b in (("oracle 10%", "stratified 10%"), ("oracle 10%", "HERMES 10%"), ("oracle 10%", "random 10%"),
                     ("oracle 10%", "unpruned"), ("oracle 5%", "stratified 5%"), ("oracle 5%", "unpruned"),
                     ("hermes_exact 10%", "HERMES 10%"), ("hermes_exact 5%", "HERMES 5%")):
            if (prompt, a) in loaded and (prompt, b) in loaded:
                A, Bq = loaded[(prompt, a)], loaded[(prompt, b)]
                keys = ("miou",) if any(r["_ok"] is None for r in list(A.values()) + list(Bq.values())) else ("acc", "miou", "gq05")
                comps[f"{a} - {b} ({prompt})"] = {k: paired(A, Bq, k) for k in keys}
    out["3b_paired_differences"] = comps
    out["5_timestamps"] = {"offset +200 s (official)": shifted("dx-offset200", 200),
                           "offset +200 s (timeline)": shifted("dx-offset200-timeline", 200),
                           "no timestamp text (official)": metrics(rows_of("dx-nostamp")) if rows_of("dx-nostamp") else None}
    out["6_primacy"] = {"start at 0.5 x question time (official)": primacy("dx-start50"),
                        "start at 0.5 x question time (timeline)": primacy("dx-start50-timeline")}
    json.dump(out, open("logs/phase15/diagnose_gpu.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
