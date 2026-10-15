"""Read-only re-scoring of CoCo paper runs (2026-10-15). Writes only into this directory.

Scorers re-implemented without their file writes:
  NExT-GQA   : hermes_pool/nextgqa/score.py (official eval_ground + TimeLens extract_time / VideoMind check_ans),
               per-question vectors as in nextgqa/paired.py.
  S-EMBER    : hermes_pool/reports/report_all.py load() with --mss (m:ss-aware), official parse otherwise.
  TimeLens   : hermes_pool/timelens/score.py convert() + evaluation/compute_metrics.py (first pair, 1 s rounding,
               failed parse -> [dur+10, dur+20]); anchors read from all.jsonl as compute_metrics does.
Bootstrap: paired, 2,000 resamples over questions, numpy default_rng(0), percentile 2.5 / 97.5.
Run with /nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3.
"""
import ast
import contextlib
import csv
import glob
import io
import json
import os
import re
import sys

import numpy as np

OUT = "/home/chieu.nguyen/.claude/jobs/7e393a16/tmp/collect2"
R = "/home/chieu.nguyen/HERMES/results/"
TLD = "/nfs-stor/chieu.nguyen/timelens-bench/"
POOL = "/home/chieu.nguyen/hermes_pool/"
sys.path.insert(0, POOL + "nextgqa/repo/code/TempGQA")
sys.path.insert(0, POOL + "timelens/repo")
sys.path.insert(0, "/home/chieu.nguyen/HERMES/eval/sember")
from eval_ground import eval_ground, get_tIoU  # noqa: E402
from timelens.utils import extract_time, iou as tl_iou  # noqa: E402
from eval_grounding import parse_grounding_response as official_parse, temporal_iou  # noqa: E402

csv.field_size_limit(10 ** 9)


# ---------------------------------------------------------------- config
KEYS = ["model_path", "batch_size", "uniform_num_frames", "sample_fps", "protocol_fps", "protocol_total_tokens",
        "protocol_min_tokens", "frame_max_pixels", "frame_min_pixels", "grounding_prompt", "prune_score",
        "offline_keep_ratio", "core_alloc", "retrieval", "contrastive_mode", "contrastive_scope", "answer_reserve",
        "repetition_penalty", "max_new_tokens", "keep_time_tokens", "anno_path"]


def config(run_dir):
    logs = sorted(glob.glob(f"{run_dir}/inference-*.log"))
    for lg in logs:
        with open(lg, errors="replace") as f:
            for line in f:
                if "Effective inference settings:" in line:
                    d = ast.literal_eval(line.split("Effective inference settings:", 1)[1].strip())
                    c = {k: d.get(k) for k in KEYS}
                    c["model_path"] = os.path.basename(str(c["model_path"]))
                    c["anno_path"] = os.path.basename(str(c["anno_path"]))
                    return c
    return None


def boot(d, n=2000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = 100 * d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    m = 100 * d.mean()
    return {"n": len(d), "delta": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2), "sig": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- NExT-GQA
NG_SUBSET = "/nfs-stor/chieu.nguyen/nextgqa/nextgqa_test_subset.jsonl"
NG_META = [json.loads(l) for l in open(NG_SUBSET)]


def check_ans(response, n_options=5):
    b = response.lower().split(" ")[0].replace("(", "").replace(")", "").replace(".", "") if response.strip() else ""
    b = b[:1]
    return b.upper() if b in [chr(ord("a") + i) for i in range(n_options)] else None


def parse_choice(reply):
    m = re.findall(r"<choice>\s*\(?([A-Ea-e])\b", reply)
    return m[-1].upper() if m else None


def parse_span(reply):
    m = re.findall(r"<answer>(.*?)</answer>", reply, flags=re.S)
    nums = re.findall(r"\d+(?:\.\d+)?", m[-1]) if m else []
    return sorted(float(x) for x in nums[:2]) if len(nums) >= 2 else None


def nextgqa(run_dir, joint=False):
    preds = {r["question_id"]: r["pred_raw"] for r in csv.DictReader(open(f"{run_dir}/results.csv"))}
    gt_ground, pred_ground, pred_qa = {}, {}, {}
    for s in NG_META:
        m, qf = s["nextgqa"], s["question_id"]
        key = f"{m['vid']}_{m['qid']}"
        gt_ground.setdefault(m["vid"], {"duration": None, "location": {}})["location"][m["qid"]] = m["spans"]
        reply = preds.get(qf.rsplit("_", 1)[0] + "_joint", "") if joint else preds.get(qf, "")
        if qf.endswith("_ans"):
            pred_qa[key] = {"answer": m["ans"], "prediction": parse_choice(reply) if joint else check_ans(reply)}
        elif joint:
            pred_ground[key] = parse_span(reply) or [-100.0, -100.0]
        else:
            ts = extract_time(reply)
            pred_ground[key] = list(ts[0]) if ts else [-100.0, -100.0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        eval_ground(gt_ground, pred_ground, pred_qa)
    official = list(map(float, out.getvalue().strip().splitlines()[-1].split()))
    names = ["Acc&GQA", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5"]
    met = {"n": len(pred_qa), "rows": len(preds),
           "Acc@QA": round(100 * sum(v["answer"] == v["prediction"] for v in pred_qa.values()) / len(pred_qa), 1),
           **dict(zip(names, official)),
           "unanswered": sum(v["prediction"] is None for v in pred_qa.values()),
           "ungrounded": sum(p == [-100.0, -100.0] for p in pred_ground.values())}
    spans = {f"{s['nextgqa']['vid']}_{s['nextgqa']['qid']}": s["nextgqa"]["spans"] for s in NG_META}
    per = {}
    for k, p in pred_ground.items():
        iou_ = max(0, max(get_tIoU(g, p)[0] for g in spans[k]))
        iop = max(0, max(get_tIoU(g, p)[1] for g in spans[k]))
        ok = pred_qa[k]["answer"] == pred_qa[k]["prediction"]
        per[k] = {"Acc&GQA": float(ok and iop >= 0.5), "Acc@QA": float(ok), "mIoP": iop, "mIoU": iou_,
                  "IoU@0.5": float(iou_ >= 0.5), "IoP@0.5": float(iop >= 0.5)}
    return met, per


# ---------------------------------------------------------------- S-EMBER (m:ss-aware)
def mss_parse(text):
    out = official_parse(text)
    if out[1] is None:
        text = re.sub(r"\b(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\b",
                      lambda m: f"{3600 * int(m.group(1)) + 60 * int(m.group(2)) + float(m.group(3)):.1f}", text)
        out = official_parse(re.sub(r"\b(\d{1,2}):(\d{1,2}(?:\.\d+)?)\b",
                                    lambda m: f"{60 * int(m.group(1)) + float(m.group(2)):.1f}", text))
    return out


def sember(path, sessions=()):
    rows = {}
    for r in csv.DictReader(open(path + "/results.csv")):
        _, st, en = mss_parse(r["pred_raw"])
        _, st0, _ = official_parse(r["pred_raw"])
        i = temporal_iou(st, en, float(r["answer_start_time"]), float(r["answer_end_time"])) if st is not None else 0.0
        rows[r["question_id"]] = {"mIoU": i, "R@0.3": float(i >= 0.3), "R@0.5": float(i >= 0.5),
                                  "R@0.7": float(i >= 0.7), "unparsed": st is None, "unparsed_official": st0 is None,
                                  "cat": r.get("question_category")}
    judged = None
    for x in sessions:
        jp = f"{path}/answer_judgments_{x}.jsonl"
        if os.path.exists(jp):
            judged = x
            for j in map(json.loads, open(jp)):
                if j["question_id"] in rows:
                    rows[j["question_id"]]["Acc"] = float(bool(j["correct"]))
                    rows[j["question_id"]]["GQ@0.5"] = float(bool(j["correct"]) and rows[j["question_id"]]["R@0.5"])
            break
    jfiles = sorted(os.path.basename(p) for p in glob.glob(path + "/answer_judgments*.jsonl"))
    met = {"n": len(rows), "unparsed": sum(r["unparsed"] for r in rows.values()),
           "unparsed_official": sum(r["unparsed_official"] for r in rows.values()), "judged_session": judged,
           "judgment_files": jfiles}
    for k in ("R@0.3", "R@0.5", "R@0.7", "mIoU", "Acc", "GQ@0.5"):
        v = [r[k] for r in rows.values() if k in r]
        met[k] = round(100 * sum(v) / len(v), 1) if v and len(v) == len(rows) else None
    cats = {}
    for r in rows.values():
        cats.setdefault(r["cat"], []).append(r["mIoU"])
    met["by_cat_mIoU"] = {c: (len(v), round(100 * sum(v) / len(v), 1)) for c, v in sorted(cats.items())}
    return met, rows


# ---------------------------------------------------------------- TimeLens
SUBSETS = {}


def subset(path):
    if path not in SUBSETS:
        SUBSETS[path] = {json.loads(l)["question_id"]: json.loads(l) for l in open(path)}
    return SUBSETS[path]


def tl_metrics(per):
    v = list(per.values())
    return {"n": len(v), "R@0.3": round(100 * np.mean([x >= 0.3 for x in v]), 2),
            "R@0.5": round(100 * np.mean([x >= 0.5 for x in v]), 2),
            "R@0.7": round(100 * np.mean([x >= 0.7 for x in v]), 2), "mIoU": round(100 * np.mean(v), 2)}


def tl_ours(run_dir, sub, time_line=False, partial=False):
    """Our pipeline (results.csv, or chunk CSVs <n>_<i>.csv if partial) -> {key: IoU}."""
    spans = subset(sub)
    files = sorted(glob.glob(f"{run_dir}/*_*.csv")) if partial else [f"{run_dir}/results.csv"]
    files = [f for f in files if re.fullmatch(r"\d+_\d+\.csv", os.path.basename(f)) or not partial]
    per, unparsed = {}, 0
    for f in files:
        for r in csv.DictReader(open(f)):
            a = spans[r["question_id"]]
            answer = r["pred_raw"]
            ts = extract_time(answer.rsplit("Time:", 1)[-1] if time_line else answer)
            if not ts:
                unparsed += 1
                ts = [[a["duration"] + 10, a["duration"] + 20]]
            ts = [[round(s), round(e)] for s, e in ts]
            key = f"{a['video_id']}.mp4>>>{a['question']}>>>{a['span']}"
            gt = eval(key.split(">>>")[2])
            per[key] = tl_iou(gt, ts[0])
    return per, unparsed, [os.path.basename(f) for f in files]


def tl_anchor(path):
    data = {}
    for line in open(path):
        data.update(json.loads(line))
    per = {}
    for key, pred in data.items():
        ts = pred["timestamps"] if "timestamps" in pred else extract_time(pred["answers"])
        if len(ts) == 0:
            ts = [(-100, -100)]
        per[key] = tl_iou(eval(key.split(">>>")[2]), ts[0])
    return per


def paired(a, b, metrics):
    """b - a on common keys; a, b: {key: {metric: value}} or {key: iou}."""
    keys = sorted(set(a) & set(b))
    out = {"n_a": len(a), "n_b": len(b), "n_common": len(keys)}
    for m in metrics:
        if m in ("tl_mIoU", "tl_R@0.5"):
            f = (lambda x: x) if m == "tl_mIoU" else (lambda x: float(x >= 0.5))
            d = [f(b[k]) - f(a[k]) for k in keys]
        else:
            d = [b[k][m] - a[k][m] for k in keys]
        out[m] = boot(d)
    return out


# ---------------------------------------------------------------- main
def main():
    res = {"configs": {}, "nextgqa": {}, "sember": {}, "tl": {}, "pairs": {}}
    # NExT-GQA
    NG = {}
    for model in ("qwen3_vl_8b", "timelens_8b"):
        for run in sorted(os.listdir(f"{R}{model}/nextgqa")):
            d = f"{R}{model}/nextgqa/{run}"
            if not os.path.exists(d + "/results.csv"):
                res["nextgqa"][f"{model}/{run}"] = {"status": "no results.csv",
                                                    "chunks": sorted(os.listdir(d))}
                continue
            met, per = nextgqa(d, joint=run.startswith("joint"))
            NG[f"{model}/{run}"] = per
            res["nextgqa"][f"{model}/{run}"] = met
            res["configs"][d] = config(d)
    ngm = ["Acc@QA", "Acc&GQA", "mIoP", "mIoU", "IoU@0.5"]
    for name, a, b in [
        ("Q3 NG Cover10 - Random10", "qwen3_vl_8b/random10", "qwen3_vl_8b/cover10"),
        ("Q3 NG Cover30 - Random30", "qwen3_vl_8b/random30", "qwen3_vl_8b/cover30"),
        ("Q3 NG Cover30 - SemVID30", "qwen3_vl_8b/semvid30", "qwen3_vl_8b/cover30"),
        ("Q3 NG Cover10 - Unpruned", "qwen3_vl_8b/unpruned", "qwen3_vl_8b/cover10"),
        ("Q3 NG Cover30 - Unpruned", "qwen3_vl_8b/unpruned", "qwen3_vl_8b/cover30"),
        ("Q3 NG Random10 - Unpruned", "qwen3_vl_8b/unpruned", "qwen3_vl_8b/random10"),
        ("Q3 NG Joint - TwoQuery (unpruned)", "qwen3_vl_8b/unpruned", "qwen3_vl_8b/joint-unpruned"),
        ("T8 NG Cover10 - Random10", "timelens_8b/random10", "timelens_8b/cover10"),
        ("T8 NG Cover10 - Unpruned", "timelens_8b/unpruned", "timelens_8b/cover10"),
        ("T8 NG Random10 - Unpruned", "timelens_8b/unpruned", "timelens_8b/random10"),
        ("T8 NG Joint - TwoQuery (unpruned)", "timelens_8b/unpruned", "timelens_8b/joint-unpruned"),
        ("NG Unpruned T8 - Q3", "qwen3_vl_8b/unpruned", "timelens_8b/unpruned"),
    ]:
        if a in NG and b in NG:
            res["pairs"][name] = paired(NG[a], NG[b], ngm)

    # S-EMBER
    SE = {}
    T8S = R + "timelens_8b/sember_grounding/"
    Q3S = R + "qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
    runs = {"T8 unpruned": T8S + "unpruned-timeline-v300", "T8 random10": T8S + "random10-timeline-v300",
            "T8 random10+PCD+ar64": T8S + "random10-pcd-ar64-timeline-v300",
            "T8 RC10": T8S + "rc64-qtopk10-timeline-v300",
            "T8 CoCo10": T8S + "rc64-qtopk10-pcd-ar64-timeline-v300",
            "Q3 unpruned": Q3S + "timeline-v300-native-time",
            "Q3 random10": Q3S + "offline-random-keep0.1-timeline-bs8-v300-native-time",
            "Q3 random10+PCD+ar64": Q3S + "p31-random10-pcd-ar64-timeline-v300-native-time",
            "Q3 RC10": Q3S + "p25-rc64-qtopk10-timeline-v300-native-time",
            "Q3 CoCo10": Q3S + "p31-rc64-qtopk10-pcd-ar64-timeline-v300-native-time"}
    for k, d in runs.items():
        met, rows = sember(d, sessions=("fresh31",) if k.startswith("Q3") else ())
        SE[k] = rows
        res["sember"][k] = {**met, "dir": d}
        res["configs"][d] = config(d)
    sm = ["mIoU", "R@0.5"]
    for m in ("T8", "Q3"):
        for name, a, b in [("RC10 - random10", "random10", "RC10"), ("CoCo10 - unpruned", "unpruned", "CoCo10"),
                           ("RC10 - unpruned", "unpruned", "RC10"), ("CoCo10 - random10", "random10", "CoCo10"),
                           ("PCD+ar64 on random10", "random10", "random10+PCD+ar64"),
                           ("PCD+ar64 on RC10", "RC10", "CoCo10"),
                           ("CoCo10 - random10+PCD+ar64", "random10+PCD+ar64", "CoCo10")]:
            extra = ["Acc", "GQ@0.5"] if m == "Q3" else []
            res["pairs"][f"{m} SE {name}"] = paired(SE[f"{m} {a}"], SE[f"{m} {b}"], sm + extra)
    for k in ("unpruned", "random10", "RC10", "CoCo10", "random10+PCD+ar64"):
        res["pairs"][f"SE T8 - Q3 {k}"] = paired(SE[f"Q3 {k}"], SE[f"T8 {k}"], sm)

    # TimeLens subset (513 QVH) and bridges (400)
    TL = {}
    QS = TLD + "qvh_tl_subset.jsonl"
    for model in ("qwen3_vl_8b", "timelens_8b"):
        for run in sorted(os.listdir(f"{R}{model}/timelens_qvh")):
            d = f"{R}{model}/timelens_qvh/{run}"
            if not os.path.exists(d + "/results.csv"):
                continue
            per, unp, _ = tl_ours(d, QS, time_line=run.startswith("tlp-"))
            TL[f"{model}/qvh/{run}"] = per
            res["tl"][f"{model}/qvh/{run}"] = {**tl_metrics(per), "unparsed": unp,
                                               "scored_on": "last Time:" if run.startswith("tlp-") else "full reply"}
            res["configs"][d] = config(d)
    for tag in ("anchor", "tl-2fps-4500", "tl-1fps-1365", "tl-1fps-5000", "t8-anchor", "t8-tl-2fps-4500",
                "t8-tl-1fps-1365"):
        p = f"{POOL}timelens/runs/{tag}/all.jsonl"
        if os.path.exists(p):
            per = tl_anchor(p)
            TL[f"script/qvh/{tag}"] = per
            res["tl"][f"script/qvh/{tag}"] = {**tl_metrics(per), "log": open(p[:-6] + ".log").read().split("\n", 1)[1]}
    for ds in ("charades", "activitynet"):
        d = f"{R}qwen3_vl_8b/timelens_{ds}/bridge-{ds}"
        per, unp, _ = tl_ours(d, f"{TLD}{ds}_bridge400.jsonl")
        TL[f"qwen3_vl_8b/{ds}/bridge"] = per
        res["tl"][f"qwen3_vl_8b/{ds}/bridge"] = {**tl_metrics(per), "unparsed": unp}
        res["configs"][d] = config(d)
    # full anchors
    for mt in ("qwen3", "t8"):
        for ds in ("charades", "qvhighlights", "activitynet"):
            p = f"{POOL}timelens/runs_full/{mt}/{ds}/anchor/all.jsonl"
            per = tl_anchor(p)
            TL[f"anchor_full/{mt}/{ds}"] = per
            res["tl"][f"anchor_full/{mt}/{ds}"] = {**tl_metrics(per),
                                                   "log": open(p[:-6] + ".log").read().split("\n", 1)[1]}
    # bridges vs anchor on same 400
    for ds in ("charades", "activitynet"):
        a = TL[f"anchor_full/qwen3/{ds}"]
        b = TL[f"qwen3_vl_8b/{ds}/bridge"]
        res["pairs"][f"TL {ds} bridge(ours unpruned) - TL script anchor (same 400)"] = paired(a, b, ["tl_mIoU"])
        res["tl"][f"anchor_on_bridge400/qwen3/{ds}"] = tl_metrics({k: a[k] for k in b if k in a})
    # full-sweep partial chunks
    for model, pre, mt in (("qwen3_vl_8b", "full-", "qwen3"), ("timelens_8b", "full-t8-", "t8")):
        for ds in ("charades", "qvhighlights", "activitynet"):
            for sel in ("cover31", "random31"):
                d = f"{R}{model}/timelens_{ds}/{pre}{ds}-{sel}"
                csvs = [f for f in glob.glob(d + "/*_*.csv") if re.fullmatch(r"\d+_\d+\.csv", os.path.basename(f))]
                key = f"full/{model}/{ds}/{sel}"
                if not csvs:
                    res["tl"][key] = {"n": 0, "chunks_done": 0}
                    continue
                per, unp, files = tl_ours(d, f"{TLD}{ds}_full.jsonl", partial=True)
                TL[key] = per
                anc = TL[f"anchor_full/{mt}/{ds}"]
                res["tl"][key] = {**tl_metrics(per), "unparsed": unp, "chunks_done": sorted(files),
                                  "anchor_same_q": tl_metrics({k: anc[k] for k in per if k in anc})}
                res["configs"][d] = config(d)
                res["pairs"][f"TL full {model} {ds} {sel}(partial) - TL script anchor unpruned"] = paired(
                    anc, per, ["tl_mIoU", "tl_R@0.5"])
    q, t = "qwen3_vl_8b/qvh/", "timelens_8b/qvh/"
    for name, a, b in [
        ("Q3 QVH513 Cover31 - Random31", q + "p300-random31", q + "p300-cover31"),
        ("Q3 QVH513 Cover31 - SemVID31", q + "p300-semvid31", q + "p300-cover31"),
        ("Q3 QVH513 Cover31 - Unpruned (ours)", q + "bridge-unpruned", q + "p300-cover31"),
        ("Q3 QVH513 Cover31 - TL script anchor", "script/qvh/anchor", q + "p300-cover31"),
        ("Q3 QVH513 Cover31 - TL script 2fps lowres", "script/qvh/tl-2fps-4500", q + "p300-cover31"),
        ("Q3 QVH513 Unpruned(ours) - TL script anchor", "script/qvh/anchor", q + "bridge-unpruned"),
        ("Q3 QVH513 RC128 Cover - Random", q + "rc128-random43", q + "rc128-cover43"),
        ("Q3 QVH513 RC64 Cover - Random", q + "rc64-random23", q + "rc64-cover23"),
        ("Q3 QVH513 RC64 Cover - TL script 1fps lowres", "script/qvh/tl-1fps-1365", q + "rc64-cover23"),
        ("Q3 QVH513 one-line: PCD(no reserve) on Cover31", q + "p300-cover31", q + "p300-cover31-pcd"),
        ("Q3 QVH513 one-line: PCD(no reserve) on RC128", q + "rc128-cover43", q + "rc128-cover43-pcd"),
        ("Q3 QVH513 one-line: PCD(no reserve) on RC64", q + "rc64-cover23", q + "rc64-cover23-pcd"),
        ("Q3 QVH513 timeline: PCD+ar64 on Cover31", q + "tlp-cover31", q + "tlp-cover31-pcd-ar64"),
        ("Q3 QVH513 timeline: PCD+ar64 on Random31", q + "tlp-random31", q + "tlp-random31-pcd-ar64"),
        ("Q3 QVH513 timeline: Cover31 - Random31", q + "tlp-random31", q + "tlp-cover31"),
        ("Q3 QVH513 timeline: Cover31+PCD+ar64 - Random31+PCD+ar64", q + "tlp-random31-pcd-ar64",
         q + "tlp-cover31-pcd-ar64"),
        ("Q3 QVH513 timeline: Cover31 - unpruned", q + "tlp-unpruned", q + "tlp-cover31"),
        ("Q3 QVH513 timeline: CoverPCDar64 - timeline unpruned", q + "tlp-unpruned", q + "tlp-cover31-pcd-ar64"),
        ("Q3 QVH513 timeline - one-line: unpruned", q + "bridge-unpruned", q + "tlp-unpruned"),
        ("Q3 QVH513 timeline - one-line: Cover31", q + "p300-cover31", q + "tlp-cover31"),
        ("Q3 QVH513 timeline - one-line: Random31", q + "p300-random31", q + "tlp-random31"),
        ("Q3 QVH513 timeline Cover+PCD+ar64 - one-line Cover", q + "p300-cover31", q + "tlp-cover31-pcd-ar64"),
        ("Q3 QVH513 timeline Cover+PCD+ar64 - one-line Cover+PCD", q + "p300-cover31-pcd", q + "tlp-cover31-pcd-ar64"),
        ("T8 QVH513 Cover31 - Random31", t + "p300-random31", t + "p300-cover31"),
        ("T8 QVH513 Cover31 - Unpruned (ours)", t + "bridge-unpruned", t + "p300-cover31"),
        ("T8 QVH513 Cover31 - TL script 2fps lowres", "script/qvh/t8-tl-2fps-4500", t + "p300-cover31"),
        ("T8 QVH513 Unpruned(ours) - TL script anchor", "script/qvh/t8-anchor", t + "bridge-unpruned"),
        ("T8 QVH513 RC64 Cover - TL script 1fps lowres", "script/qvh/t8-tl-1fps-1365", t + "rc64-cover23"),
        ("T8 QVH513 RC64 Cover - Unpruned (ours)", t + "bridge-unpruned", t + "rc64-cover23"),
    ]:
        if a in TL and b in TL:
            res["pairs"][name] = paired(TL[a], TL[b], ["tl_mIoU", "tl_R@0.5"])
        else:
            res["pairs"][name] = {"missing": [x for x in (a, b) if x not in TL]}
    json.dump(res, open(f"{OUT}/collect.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
