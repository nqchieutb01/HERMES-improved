"""Read-only scoring of the full TimeLens-Bench sweep (Qwen3-VL-8B + TimeLens-8B dirs), Cover31 vs Random31.
Scoring copied from docs/paper/analysis/collect_2026-10-15.py (tl_ours / tl_anchor / boot)."""
import ast, csv, glob, json, os, re, sys
import numpy as np
OUT = "/home/chieu.nguyen/.claude/jobs/7e393a16/tmp/collect3"
R = "/home/chieu.nguyen/HERMES/results/"
TLD = "/nfs-stor/chieu.nguyen/timelens-bench/"
POOL = "/home/chieu.nguyen/hermes_pool/"
sys.path.insert(0, POOL + "timelens/repo")
from timelens.utils import extract_time, iou as tl_iou  # noqa: E402
csv.field_size_limit(10 ** 9)
EXPECT = {"charades": 3363, "qvhighlights": 1541, "activitynet": 4500}
NCH = {"charades": 12, "qvhighlights": 6, "activitynet": 18}
CFGK = ("prune_score", "offline_keep_ratio", "protocol_fps", "protocol_total_tokens", "protocol_min_tokens",
        "frame_max_pixels", "grounding_prompt", "uniform_num_frames", "sample_fps", "num_chunks", "anno_path",
        "model_path", "answer_reserve", "contrastive_mode", "prune_seed", "keep_time_tokens")


def boot(d, n=2000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = 100 * d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(d), "delta": round(100 * d.mean(), 2), "lo": round(lo, 2), "hi": round(hi, 2),
            "sig": bool(lo > 0 or hi < 0)}


def mets(v):
    v = np.asarray(v, float)
    return {"n": len(v), "R@0.3": round(100 * np.mean(v >= .3), 2), "R@0.5": round(100 * np.mean(v >= .5), 2),
            "R@0.7": round(100 * np.mean(v >= .7), 2), "mIoU": round(100 * np.mean(v), 2)}


def score_dir(d, sub):
    spans = {}
    for l in open(sub):
        x = json.loads(l)
        spans[x["question_id"]] = x
    files = sorted(f for f in glob.glob(f"{d}/*_*.csv") if re.fullmatch(r"\d+_\d+\.csv", os.path.basename(f)))
    per, raw, dup_same, dup_diff, unparsed, rows = {}, {}, 0, 0, 0, 0
    for f in files:
        for r in csv.DictReader(open(f)):
            rows += 1
            qid = r["question_id"]
            if qid in raw:
                if raw[qid] == r["pred_raw"]:
                    dup_same += 1
                else:
                    dup_diff += 1
                continue  # keep first occurrence
            raw[qid] = r["pred_raw"]
            a = spans[qid]
            ts = extract_time(r["pred_raw"])
            if not ts:
                unparsed += 1
                ts = [[a["duration"] + 10, a["duration"] + 20]]
            ts = [[round(s), round(e)] for s, e in ts]
            key = f"{a['video_id']}.mp4>>>{a['question']}>>>{a['span']}"
            per[qid] = (key, tl_iou(eval(key.split(">>>")[2]), ts[0]))
    keys = [k for k, _ in per.values()]
    return per, {"files": [os.path.basename(f) for f in files], "rows": rows, "dup_same": dup_same,
                 "dup_diff": dup_diff, "unparsed": unparsed, "n_sub": len(spans),
                 "distinct_keys": len(set(keys))}


def anchor(p):
    data = {}
    for line in open(p):
        data.update(json.loads(line))
    per = {}
    for key, pred in data.items():
        ts = pred["timestamps"] if "timestamps" in pred else extract_time(pred["answers"])
        if len(ts) == 0:
            ts = [(-100, -100)]
        per[key] = tl_iou(eval(key.split(">>>")[2]), ts[0])
    return per


def logs(d):
    bs, cfgs, kept, tot, nlog = set(), set(), [], [], 0
    for lg in sorted(glob.glob(f"{d}/inference-*.log")):
        nlog += 1
        for line in open(lg, errors="replace"):
            if "Effective inference settings:" in line:
                c = ast.literal_eval(line.split("Effective inference settings:", 1)[1].strip())
                bs.add(c["batch_size"])
                cfgs.add(tuple((k, str(c.get(k))) for k in CFGK))
            m = re.search(r"Offline pruning: keeping (\d+) of (\d+) visual tokens", line)
            if m:
                kept.append(int(m.group(1)))
                tot.append(int(m.group(2)))
    return {"n_logs": nlog, "batch_size": sorted(bs), "configs": [dict(c) for c in cfgs],
            "kept_mean": round(float(np.mean(kept)), 1) if kept else None,
            "total_mean": round(float(np.mean(tot)), 1) if tot else None,
            "kept_ratio": round(float(np.sum(kept) / np.sum(tot)), 4) if kept else None,
            "kept_min": min(kept) if kept else None, "kept_max": max(kept) if kept else None,
            "n_prune_lines": len(kept)}


res = {"cells": {}, "pairs": {}, "anchors": {}}
for model, pre, mt in (("qwen3_vl_8b", "full-", "qwen3"), ("timelens_8b", "full-t8-", "t8")):
    for ds in ("charades", "qvhighlights", "activitynet"):
        sub = f"{TLD}{ds}_full.jsonl"
        anc_p = f"{POOL}timelens/runs_full/{mt}/{ds}/anchor/all.jsonl"
        anc = anchor(anc_p) if os.path.exists(anc_p) else None
        res["anchors"][f"{mt}/{ds}"] = mets(list(anc.values())) if anc else None
        cells = {}
        for sel in ("cover31", "random31"):
            d = f"{R}{model}/timelens_{ds}/{pre}{ds}-{sel}"
            per, info = score_dir(d, sub) if os.path.isdir(d) else ({}, {"files": []})
            if os.path.isdir(d):
                info.update(logs(d))
            info["expected"] = EXPECT[ds]
            info["n_chunks"] = NCH[ds]
            if per:
                info.update(mets([v for _, v in per.values()]))
                if anc:
                    info["missing_in_anchor"] = sum(k not in anc for k, _ in per.values())
                    info["anchor_same_q"] = mets([anc[k] for k, _ in per.values() if k in anc])
            else:
                info["n"] = 0
            res["cells"][f"{model}/{ds}/{sel}"] = info
            cells[sel] = per
        if cells["cover31"] and cells["random31"] and anc:
            q = sorted(set(cells["cover31"]) & set(cells["random31"]))
            q = [x for x in q if cells["cover31"][x][0] in anc]
            c = np.array([cells["cover31"][x][1] for x in q])
            r = np.array([cells["random31"][x][1] for x in q])
            u = np.array([anc[cells["cover31"][x][0]] for x in q])
            out = {"n": len(q), "Cover": mets(c), "Random": mets(r), "Unpruned": mets(u)}
            for name, a, b in (("Cover-Random", c, r), ("Cover-Unpruned", c, u), ("Random-Unpruned", r, u)):
                out[name] = {m: boot((a >= t).astype(float) - (b >= t).astype(float))
                             for m, t in (("R@0.3", .3), ("R@0.5", .5), ("R@0.7", .7))}
                out[name]["mIoU"] = boot(a - b)
            res["pairs"][f"{model}/{ds}"] = out
json.dump(res, open(f"{OUT}/full_sweep.json", "w"), indent=1, default=str)
for k, v in res["cells"].items():
    print(k, {x: v.get(x) for x in ("n", "expected", "rows", "dup_same", "dup_diff", "unparsed", "distinct_keys",
                                     "R@0.3", "R@0.5", "R@0.7", "mIoU", "batch_size", "kept_mean", "total_mean",
                                     "kept_ratio", "kept_min", "kept_max", "n_prune_lines", "n_logs",
                                     "missing_in_anchor")}, v.get("files"), "\n  anchor_same_q", v.get("anchor_same_q"),
          "\n  configs", v.get("configs"))
print("ANCHORS", res["anchors"])
print(json.dumps(res["pairs"], indent=1))
