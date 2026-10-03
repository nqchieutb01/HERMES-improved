"""Results for prior-contrastive decoding (PCD) and self-window zoom on S-EMBER grounding (Qwen3-VL-8B, 475 questions).

Each block pairs a method run with its matched baseline (same frames, selector and budget; only the method differs)
and reports Acc. (official S-EMBER judge prompt), mIoU, R@0.5 and GQ@0.5, the paired difference with a 95% bootstrap
interval (* = excludes zero), and the prior-leakage rates: answers whose first "Seen" moment is at 0 s and intervals
starting in the first 5% of the window. Runs that are not finished or judged are listed as pending.
Writes logs/phase18/report.md. Run from the repo root: python3 logs/phase18/report.py
"""
import re
import sys

sys.path.insert(0, "logs/phase15")
from diagnose_gpu import metrics, paired, rows_of  # noqa: E402

SEEN = re.compile(r"^\s*Seen:\s*([\d.]+)", re.M)
S = "time-count-location-fps0.2-kv{}-k0-attention_weighted-native-time-keepsurv-{}v300"
BLOCKS = [
    ("PCD across budgets (random selector, timeline prompt)", [
        ("5%", "offline-random-keep0.05-timeline", "random5-cd-blind-a0.5-timeline"),
        ("10%", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
        ("25%", "offline-random-keep0.25-timeline", "random25-cd-blind-a0.5-timeline"),
        ("100% (unpruned)", "timeline", "unpruned-cd-blind-a0.5-timeline")]),
    ("PCD across selectors (10%, timeline prompt)", [
        ("HERMES", "offline-hermes-keep0.1-timeline", "hermes10-cd-blind-a0.5-timeline"),
        ("Stratified", "offline-stratified-keep0.1-timeline", "stratified10-cd-blind-a0.5-timeline"),
        ("Random", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline")]),
    ("PCD in HERMES streaming (0.2 fps, timeline prompt)", [
        ("KV 6k", "@" + S.format(6000, "timeline-"), "@" + S.format(6000, "cd-blind-a0.5-timeline-")),
        ("KV 4k", "@" + S.format(4000, "timeline-"), "@" + S.format(4000, "cd-blind-a0.5-timeline-"))]),
    ("PCD with the official prompt", [
        ("Random 10%", "offline-random-keep0.1", "random10-cd-blind-a0.5"),
        ("Unpruned", "", "unpruned-cd-blind-a0.5")]),
    ("PCD hyperparameters (random 10%, timeline prompt)", [
        ("alpha 0.25", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.25-timeline"),
        ("alpha 0.5", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
        ("alpha 1.0", "offline-random-keep0.1-timeline", "random10-cd-blind-a1.0-timeline"),
        ("alpha 0.5, beta 0.2", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-b0.2-timeline")]),
    ("Counterfactual memory for the contrast (random 10%, timeline prompt)", [
        ("no video (PCD)", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
        ("permuted timestamps, alpha 0.5", "offline-random-keep0.1-timeline", "random10-cd-stamps-a0.5-timeline"),
        ("permuted timestamps, alpha 1", "offline-random-keep0.1-timeline", "random10-cd-stamps-a1-timeline"),
        ("visual-attention gain 4 (no contrast)", "offline-random-keep0.1-timeline", "random10-vag4-timeline")]),
    ("Time-scoped PCD: contrast only on time values (timeline prompt)", [
        ("Random 10%, all tokens", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
        ("Random 10%, time", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-scopetime-timeline"),
        ("Random 10%, time + answer", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-scopetime_answer-timeline"),
        ("Random 25%, time", "offline-random-keep0.25-timeline", "random25-cd-blind-a0.5-scopetime-timeline"),
        ("Unpruned, time", "timeline", "unpruned-cd-blind-a0.5-scopetime-timeline"),
        ("HERMES 10%, time", "offline-hermes-keep0.1-timeline", "hermes10-cd-blind-a0.5-scopetime-timeline"),
        ("Streaming KV 4k, time", "@" + S.format(4000, "timeline-"), "@" + S.format(4000, "cd-blind-a0.5-scopetime-timeline-")),
        ("Random 10%, time, official prompt", "offline-random-keep0.1", "random10-cd-blind-a0.5-scopetime")]),
    ("VCD baseline: noised frames as the counterfactual (random 10%, timeline prompt)", [
        ("no video (PCD), all tokens", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
        ("noised frames (VCD), all tokens", "offline-random-keep0.1-timeline", "random10-cd-noise-a0.5-scopeall-timeline"),
        ("noised frames (VCD), time", "offline-random-keep0.1-timeline", "random10-cd-noise-a0.5-scopetime-timeline")]),
    ("Self-window zoom and its combination with PCD (timeline prompt)", [
        ("zoom, 10%", "offline-random-keep0.1-timeline", "zoom-self-keep0.1-k32-a0.6-timeline"),
        ("zoom + PCD, 10%", "offline-random-keep0.1-timeline", "zoom-self-keep0.1-k32-a0.6-cd-blind-a0.5-timeline"),
        ("zoom, 5%", "offline-random-keep0.05-timeline", "zoom-self-keep0.05-k32-a0.6-timeline"),
        ("zoom + PCD, 5%", "offline-random-keep0.05-timeline", "zoom-self-keep0.05-k32-a0.6-cd-blind-a0.5-timeline")]),
    ("Zoom ablations (10%, random coarse pass, timeline prompt unless noted)", [
        ("whole-video window (control)", "offline-random-keep0.1-timeline", "zoom-full-keep0.1-k32-a0.6-timeline"),
        ("gold window", "offline-random-keep0.1-timeline", "zoom-gold-keep0.1-k32-a0.6-timeline"),
        ("gold window + same margin", "offline-random-keep0.1-timeline", "zoom-goldmargin-keep0.1-k32-a0.6-timeline"),
        ("self, margin 0.5x/10 s", "offline-random-keep0.1-timeline", "zoom-self-m0.5-keep0.1-k32-a0.6-timeline"),
        ("self, margin 2x/30 s", "offline-random-keep0.1-timeline", "zoom-self-m2-keep0.1-k32-a0.6-timeline"),
        ("self, share 0.4", "offline-random-keep0.1-timeline", "zoom-self-keep0.1-k32-a0.4-timeline"),
        ("self, share 0.8", "offline-random-keep0.1-timeline", "zoom-self-keep0.1-k32-a0.8-timeline"),
        ("self, stratified coarse pass", "offline-stratified-keep0.1-timeline", "zoom-self-strat-keep0.1-k32-a0.6-timeline"),
        ("self, official prompt", "offline-random-keep0.1", "zoom-self-official-keep0.1-k32-a0.6"),
        ("self, 25%", "offline-random-keep0.25-timeline", "zoom-self-keep0.25-k32-a0.6-timeline")]),
]


def load(tag):
    if tag.startswith("@"):  # streaming run directory (not under the uniform-n64 prefix)
        return _stream(tag[1:])
    rows = rows_of(tag)
    return rows if rows and all(r["_ok"] is not None for r in rows.values()) else None


def _stream(name):
    import json
    import os
    d = f"results/qwen3_vl_8b/sember_grounding/{name}"
    if not (os.path.exists(f"{d}/sember_grounding_scored.jsonl") and os.path.exists(f"{d}/answer_judgments.jsonl")):
        return None
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{d}/sember_grounding_scored.jsonl")}
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{d}/answer_judgments.jsonl")}
    for q, r in rows.items():
        r["_ok"] = judge.get(q)
        r["_iou"] = float(r["temporal_iou"] or 0)
        r["_gs"], r["_ge"], r["_qt"] = float(r["answer_start_time"]), float(r["answer_end_time"]), float(r["question_time"])
    return rows


def leakage(rows):
    v = list(rows.values())
    seen = [SEEN.findall(r["pred_raw"] or "") for r in v]
    zero = [float(s[0]) <= 0.5 for s in seen if s]
    p = [r for r in v if r["interval_parseable"]]
    early = [min(float(r["pred_start_time"]), float(r["pred_end_time"])) < 0.05 * r["_qt"] for r in p]
    return (100 * sum(zero) / len(zero) if zero else None), 100 * sum(early) / max(len(early), 1)


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def main():
    out = ["# Prior-contrastive decoding and self-window zoom: results", "",
           "Generated by `logs/phase18/report.py`. Each method row is paired with its matched baseline (same frames, "
           "selector and budget). Δ = method − baseline with a 95% paired bootstrap interval; * = interval excludes 0. "
           "Leak: first `Seen` moment at 0 s / interval starting in the first 5% of the window (timeline prompt).", ""]
    for title, rows in BLOCKS:
        out += [f"## {title}", "",
                "| Setting | Run | Acc. | mIoU | R@0.5 | GQ@0.5 | ΔAcc. | ΔmIoU | ΔGQ@0.5 | Leak 0 s / early |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for label, base_tag, method_tag in rows:
            base, meth = load(base_tag), load(method_tag)
            if base is None:
                out.append(f"| {label} | baseline | pending | | | | | | | |")
                continue
            bm, bl = metrics(base), leakage(base)
            out.append(f"| {label} | baseline | {bm['acc']} | {bm['miou']} | {bm['r05']} | {bm['gq05']} | | | | "
                       f"{fmt(bl[0])} / {fmt(bl[1])} |")
            if meth is None:
                out.append(f"| | method | pending | | | | | | | |")
                continue
            mm, ml = metrics(meth), leakage(meth)
            d = [paired(meth, base, k) for k in ("acc", "miou", "gq05")]
            out.append(f"| | method | {mm['acc']} | {mm['miou']} | {mm['r05']} | {mm['gq05']} | {d[0]} | {d[1]} | {d[2]} | "
                       f"{fmt(ml[0])} / {fmt(ml[1])} |")
        out.append("")
    open("logs/phase18/report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
