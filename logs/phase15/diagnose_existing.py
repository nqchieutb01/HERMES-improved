"""Phase 15: where, how and why grounded QA fails, from existing S-EMBER grounding outputs (no GPU).

Qwen3-VL-8B, 475 grounding questions. Sections:
  A. WHERE: accuracy, mIoU and GQ@0.5 by category, question time, memory recency (question time minus
     gold end), gold length, gold position; per configuration.
  B. HOW: interval error taxonomy (unparsed, disjoint before/after, inside-too-short, covers-too-long,
     partial overlap) and signed start/end errors.
  C. ANSWER vs INTERVAL: for duration questions, stated duration vs predicted interval length vs gold
     duration; for counting, predicted vs gold count (over/under).
  D. SEEN MOMENTS (timeline prompt): share of "Seen" timestamps inside the gold interval, questions with
     no Seen inside gold, first Seen at 0 s, and Seen timestamps that match no timestamp shown to the model.
  E. PRUNING-INDUCED vs INTRINSIC: questions right unpruned but wrong pruned, and questions no run gets right.
  F. SAMPLING LIMITS: uniform frames inside the gold interval, and accuracy/IoU by that count.
  G. STREAMING RETENTION: tokens kept for gold-interval frames vs other frames at answer time.
Writes logs/phase15/diagnose_existing.json and prints the tables. Run from the repo root:
python3 logs/phase15/diagnose_existing.py
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, "logs/phase11")
import rule_answer as ra  # noqa: E402

B = "results/qwen3_vl_8b/sember_grounding/"
U = "uniform-n64-time-count-location-"
S = "time-count-location-fps0.2-kv{}-k0-attention_weighted-native-time-keepsurv-{}v300"
CONFIGS = {
    "uni64 official": U + "v300-native-time",
    "uni64 timeline": U + "timeline-v300-native-time",
    "HERMES10 official": U + "offline-hermes-keep0.1-v300-native-time",
    "HERMES10 timeline": U + "offline-hermes-keep0.1-timeline-v300-native-time",
    "strat10 official": U + "offline-stratified-keep0.1-v300-native-time",
    "strat10 timeline": U + "offline-stratified-keep0.1-timeline-v300-native-time",
    "random10 official": U + "offline-random-keep0.1-v300-native-time",
    "random10 timeline": U + "offline-random-keep0.1-timeline-v300-native-time",
    "HERMES5 official": U + "offline-hermes-keep0.05-v300-native-time",
    "HERMES5 timeline": U + "offline-hermes-keep0.05-timeline-v300-native-time",
    "stream6k official": S.format(6000, ""),
    "stream6k timeline": S.format(6000, "timeline-"),
}
OUT = {}


def load(tag):
    d = B + tag
    rows = [json.loads(l) for l in open(f"{d}/sember_grounding_scored.jsonl")]
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{d}/answer_judgments.jsonl")}
    for r in rows:
        r["_ok"] = judge[r["question_id"]]
        r["_iou"] = float(r["temporal_iou"] or 0)
        r["_gs"], r["_ge"], r["_qt"] = float(r["answer_start_time"]), float(r["answer_end_time"]), float(r["question_time"])
        if r["interval_parseable"]:
            r["_ps"], r["_pe"] = sorted((float(r["pred_start_time"]), float(r["pred_end_time"])))
        else:
            r["_ps"] = r["_pe"] = None
    return rows


def summary(rows):
    n = len(rows)
    if not n:
        return None
    return {"n": n, "acc": round(100 * sum(r["_ok"] for r in rows) / n, 1),
            "miou": round(100 * sum(r["_iou"] for r in rows) / n, 1),
            "gq05": round(100 * sum(r["_ok"] and r["_iou"] >= 0.5 for r in rows) / n, 1)}


def binned(rows, key, edges, labels):
    out = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        out[lab] = summary([r for r in rows if lo <= key(r) < hi])
    return out


def section_where(data):
    res = {}
    dims = {
        "category": lambda rows: {c: summary([r for r in rows if r["question_category"] == c])
                                  for c in ("time_duration", "counting_objects_events", "location_trace")},
        "question_time": lambda rows: binned(rows, lambda r: r["_qt"], [0, 80, 160, 320, 1e9],
                                             ["<80s", "80-160s", "160-320s", ">=320s"]),
        "recency (qtime - gold end)": lambda rows: binned(rows, lambda r: r["_qt"] - r["_ge"], [-1e9, 10, 60, 180, 1e9],
                                                          ["<10s", "10-60s", "60-180s", ">=180s"]),
        "gold length": lambda rows: binned(rows, lambda r: r["_ge"] - r["_gs"], [0, 15, 45, 120, 1e9],
                                           ["<15s", "15-45s", "45-120s", ">=120s"]),
        "gold start / qtime": lambda rows: binned(rows, lambda r: r["_gs"] / max(r["_qt"], 1), [0, 0.25, 0.5, 0.75, 10],
                                                  ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]),
    }
    for dim, fn in dims.items():
        res[dim] = {cfg: fn(rows) for cfg, rows in data.items()}
    return res


def taxonomy(r):
    if r["_ps"] is None:
        return "unparsed"
    ps, pe, gs, ge = r["_ps"], r["_pe"], r["_gs"], r["_ge"]
    if pe < gs:
        return "disjoint, before gold"
    if ps > ge:
        return "disjoint, after gold"
    if ps >= gs and pe <= ge:
        return "inside gold (too short)"
    if ps <= gs and pe >= ge:
        return "covers gold (too long)"
    return "partial overlap"


def section_how(data):
    res = {}
    for cfg, rows in data.items():
        c = collections.Counter(taxonomy(r) for r in rows)
        p = [r for r in rows if r["_ps"] is not None]
        res[cfg] = {
            "taxonomy_%": {k: round(100 * v / len(rows), 1) for k, v in c.most_common()},
            "median start error s (pred - gold)": round(st.median(r["_ps"] - r["_gs"] for r in p), 1),
            "median end error s": round(st.median(r["_pe"] - r["_ge"] for r in p), 1),
            "median length ratio pred/gold": round(st.median((r["_pe"] - r["_ps"]) / max(r["_ge"] - r["_gs"], 1) for r in p), 2),
            "IoU>=0.5 %": round(100 * sum(r["_iou"] >= 0.5 for r in rows) / len(rows), 1),
            "answer right but IoU<0.1 %": round(100 * sum(r["_ok"] and r["_iou"] < 0.1 for r in rows) / len(rows), 1),
            "answer wrong but IoU>=0.5 %": round(100 * sum((not r["_ok"]) and r["_iou"] >= 0.5 for r in rows) / len(rows), 1),
        }
    return res


def section_answer_interval(data):
    res = {}
    for cfg, rows in data.items():
        dur = []
        for r in rows:
            if r["question_category"] != "time_duration":
                continue
            said = ra.parse_duration(ra.model_answer(r))
            golds = [ra.parse_duration(a) for a in ra.annotator_answers(r)]
            golds = [g for g in golds if g]
            if said is None or not golds:
                continue
            length = (r["_pe"] - r["_ps"]) if r["_ps"] is not None else None
            dur.append((said, st.median(golds), length, r["_ge"] - r["_gs"]))
        cnt = []
        for r in rows:
            if r["question_category"] != "counting_objects_events":
                continue
            said = ra.parse_count(ra.model_answer(r))
            golds = [ra.parse_count(a) for a in ra.annotator_answers(r)]
            golds = [g for g in golds if g is not None]
            if said is not None and golds:
                cnt.append((said, st.median(golds)))
        consistent = [d for d in dur if d[2] is not None and d[2] > 0]
        res[cfg] = {
            "duration: n parsed": len(dur),
            "duration: median stated / gold": round(st.median(d[0] / max(d[1], 1) for d in dur), 2) if dur else None,
            "duration: stated within 25% of gold %": round(100 * sum(abs(d[0] - d[1]) <= 0.25 * d[1] for d in dur) / len(dur), 1) if dur else None,
            "duration: stated within 25% of own interval length %": round(100 * sum(abs(d[0] - d[2]) <= 0.25 * max(d[2], 1) for d in consistent) / len(consistent), 1) if consistent else None,
            "duration: gold answer vs gold interval length, median ratio": round(st.median(d[1] / max(d[3], 1) for d in dur), 2) if dur else None,
            "count: n parsed": len(cnt),
            "count: exact %": round(100 * sum(c[0] == c[1] for c in cnt) / len(cnt), 1) if cnt else None,
            "count: under %": round(100 * sum(c[0] < c[1] for c in cnt) / len(cnt), 1) if cnt else None,
            "count: over %": round(100 * sum(c[0] > c[1] for c in cnt) / len(cnt), 1) if cnt else None,
            "count: median gold": st.median(c[1] for c in cnt) if cnt else None,
            "count: median predicted": st.median(c[0] for c in cnt) if cnt else None,
        }
    return res


SEEN = re.compile(r"^\s*Seen:\s*([\d.]+)\s*(?:s|sec|seconds)?", re.M)


def shown_stamps(r):
    """Timestamps shown to the model: Qwen3 stamps each frame pair with the pair's mean time."""
    if r.get("frame_sampling") == "uniform":
        idx = json.loads(r["source_frame_indices_json"])
        fps = idx[-1] / max(r["_qt"], 1e-6) if idx[-1] else 1.0
        t = [i / fps for i in idx]
    else:
        t = [5.0 * i for i in range(int(r["_qt"] // 5) + 1)]
    return [(t[i] + t[min(i + 1, len(t) - 1)]) / 2 for i in range(0, len(t), 2)]


def section_seen(data):
    res = {}
    for cfg, rows in data.items():
        if "timeline" not in cfg:
            continue
        per_q = []
        for r in rows:
            seen = [float(x) for x in SEEN.findall(r["pred_raw"] or "")]
            if not seen:
                per_q.append(None)
                continue
            stamps = shown_stamps(r)
            inside = [r["_gs"] - 2.5 <= t <= r["_ge"] + 2.5 for t in seen]
            valid = [min(abs(t - s) for s in stamps) <= 0.6 for t in seen] if stamps else [True] * len(seen)
            per_q.append({"n": len(seen), "inside": sum(inside), "first0": seen[0] <= 1.0, "valid": sum(valid),
                          "ok": r["_ok"], "iou": r["_iou"], "cat": r["question_category"]})
        q = [x for x in per_q if x]
        tot = sum(x["n"] for x in q)
        hit = [x for x in q if x["inside"] > 0]
        miss = [x for x in q if x["inside"] == 0]
        res[cfg] = {
            "questions with a Seen list %": round(100 * len(q) / len(rows), 1),
            "median # Seen": st.median(x["n"] for x in q),
            "Seen inside gold (+-2.5s) %": round(100 * sum(x["inside"] for x in q) / tot, 1),
            "questions with NO Seen inside gold %": round(100 * len(miss) / len(q), 1),
            "first Seen at ~0 s %": round(100 * sum(x["first0"] for x in q) / len(q), 1),
            "Seen matching a shown timestamp %": round(100 * sum(x["valid"] for x in q) / tot, 1),
            "acc when some Seen inside gold": round(100 * sum(x["ok"] for x in hit) / max(len(hit), 1), 1),
            "acc when no Seen inside gold": round(100 * sum(x["ok"] for x in miss) / max(len(miss), 1), 1),
            "mIoU when some Seen inside gold": round(100 * sum(x["iou"] for x in hit) / max(len(hit), 1), 1),
            "mIoU when no Seen inside gold": round(100 * sum(x["iou"] for x in miss) / max(len(miss), 1), 1),
        }
    return res


def section_induced(data):
    res = {}
    base = {r["question_id"]: r for r in data["uni64 timeline"]}
    base_o = {r["question_id"]: r for r in data["uni64 official"]}
    for cfg, rows in data.items():
        if cfg.startswith("uni64"):
            continue
        ref = base if "timeline" in cfg else base_o
        by = {r["question_id"]: r for r in rows}
        qs = sorted(set(by) & set(ref))
        lost = [q for q in qs if ref[q]["_ok"] and not by[q]["_ok"]]
        gained = [q for q in qs if by[q]["_ok"] and not ref[q]["_ok"]]
        g_lost = [q for q in qs if ref[q]["_iou"] >= 0.5 and by[q]["_iou"] < 0.5]
        g_gain = [q for q in qs if by[q]["_iou"] >= 0.5 and ref[q]["_iou"] < 0.5]
        res[cfg] = {"answers lost vs unpruned (same prompt)": len(lost), "answers gained": len(gained),
                    "IoU>=0.5 lost": len(g_lost), "IoU>=0.5 gained": len(g_gain)}
    # Questions no run answers correctly, over every judged Qwen3 grounding run.
    ever_ok, ever_g = collections.Counter(), collections.Counter()
    runs = 0
    for jpath in glob.glob(B + "*/answer_judgments.jsonl"):
        d = os.path.dirname(jpath)
        if not os.path.exists(f"{d}/sember_grounding_scored.jsonl"):
            continue
        runs += 1
        judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(jpath)}
        for l in open(f"{d}/sember_grounding_scored.jsonl"):
            r = json.loads(l)
            ever_ok[r["question_id"]] += judge.get(r["question_id"], False)
            ever_g[r["question_id"]] += judge.get(r["question_id"], False) and float(r["temporal_iou"] or 0) >= 0.5
    qs = [r["question_id"] for r in data["uni64 official"]]
    cats = {r["question_id"]: r["question_category"] for r in data["uni64 official"]}
    never = [q for q in qs if ever_ok[q] == 0]
    never_g = [q for q in qs if ever_g[q] == 0]
    res["across all judged runs"] = {
        "runs": runs,
        "questions never answered correctly %": round(100 * len(never) / len(qs), 1),
        "never answered, by category": dict(collections.Counter(cats[q] for q in never)),
        "questions never GQ@0.5 %": round(100 * len(never_g) / len(qs), 1),
        "questions answered correctly by >=half of runs %": round(100 * sum(ever_ok[q] >= runs / 2 for q in qs) / len(qs), 1),
    }
    return res


def section_sampling(data):
    rows = data["uni64 official"]
    tl = {r["question_id"]: r for r in data["uni64 timeline"]}
    buckets = collections.defaultdict(list)
    for r in rows:
        idx = json.loads(r["source_frame_indices_json"])
        fps = idx[-1] / max(r["_qt"], 1e-6) if idx[-1] else 1.0
        n_in = sum(r["_gs"] <= i / fps <= r["_ge"] for i in idx)
        lab = "0" if n_in == 0 else "1" if n_in == 1 else "2-3" if n_in <= 3 else "4-8" if n_in <= 8 else ">8"
        buckets[lab].append((r, tl.get(r["question_id"])))
    res = {}
    for lab in ("0", "1", "2-3", "4-8", ">8"):
        pairs = buckets.get(lab, [])
        if pairs:
            res[f"{lab} frames in gold"] = {"official": summary([p[0] for p in pairs]),
                                            "timeline": summary([p[1] for p in pairs if p[1]])}
    return res


def section_retention():
    res = {}
    for tag in glob.glob(B + "*-retention-v300"):
        name = os.path.basename(tag).replace("time-count-location-", "").replace("-attention_weighted-native-time-keepsurv-retention-v300", "")
        judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{tag}/answer_judgments.jsonl")} \
            if os.path.exists(f"{tag}/answer_judgments.jsonl") else {}
        scored = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{tag}/sember_grounding_scored.jsonl")}
        stats = []
        for f in glob.glob(f"{tag}/retention-*.jsonl"):
            for l in open(f):
                s = json.loads(l)
                r = scored.get(s["question_id"])
                if not r:
                    continue
                gs, ge = float(r["answer_start_time"]), float(r["answer_end_time"])
                times = s["frame_times"]
                enc = {int(k): v for k, v in s["encoded"].items()}
                kept = {int(k): v for k, v in s["kept"].items()}
                gold = [i for i, t in enumerate(times) if gs <= t <= ge]
                other = [i for i in range(len(times)) if i not in set(gold)]
                share = lambda ids: sum(kept.get(i, 0) for i in ids) / max(sum(enc.get(i, 0) for i in ids), 1)
                dead = lambda ids: sum(kept.get(i, 0) < 1 for i in ids) / max(len(ids), 1)
                if gold and other:
                    stats.append({"gold_keep": share(gold), "other_keep": share(other), "gold_dead": dead(gold),
                                  "other_dead": dead(other), "ok": judge.get(s["question_id"]),
                                  "iou": float(r["temporal_iou"] or 0)})
        if not stats:
            continue
        good = [x for x in stats if x["iou"] >= 0.3]
        bad = [x for x in stats if x["iou"] < 0.3]
        m = lambda xs, k: round(100 * st.mean(x[k] for x in xs), 1) if xs else None
        res[name] = {"questions": len(stats),
                     "gold frames: % tokens kept": m(stats, "gold_keep"), "other frames: % tokens kept": m(stats, "other_keep"),
                     "gold frames with no token %": m(stats, "gold_dead"), "other frames with no token %": m(stats, "other_dead"),
                     "gold % kept when IoU>=0.3": m(good, "gold_keep"), "gold % kept when IoU<0.3": m(bad, "gold_keep")}
    return res


def show(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=1))


def main():
    data = {cfg: load(tag) for cfg, tag in CONFIGS.items() if os.path.exists(B + tag + "/answer_judgments.jsonl")}
    OUT["A_where"] = section_where(data)
    OUT["B_how"] = section_how(data)
    OUT["C_answer_vs_interval"] = section_answer_interval(data)
    OUT["D_seen_moments"] = section_seen(data)
    OUT["E_pruning_induced"] = section_induced(data)
    OUT["F_sampling_limits"] = section_sampling(data)
    OUT["G_streaming_retention"] = section_retention()
    json.dump(OUT, open("logs/phase15/diagnose_existing.json", "w"), indent=1)
    for k in ("B_how", "C_answer_vs_interval", "D_seen_moments", "E_pruning_induced", "F_sampling_limits",
              "G_streaming_retention"):
        show(k, OUT[k])


if __name__ == "__main__":
    main()
