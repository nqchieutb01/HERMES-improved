"""CVPR LaTeX tables for prior-contrastive decoding (PCD), Qwen3-VL-8B on S-EMBER grounded QA (475 questions, 300 videos).

tables/pcd_main.tex     : baseline vs +PCD across token budgets, selectors and HERMES streaming (timeline prompt).
tables/pcd_ablation.tex : counterfactual memory, strength/cut-off, prompt, and alternative methods (random 10%).
Gains are paired against the same memory; bold marks a 95% paired bootstrap interval that excludes zero.
Run from the repo root: python3 tables/make_pcd_tables.py
"""
import sys

sys.path.insert(0, "logs/phase18")
sys.path.insert(0, "logs/phase15")
from diagnose_gpu import boot  # noqa: E402
from report import leakage, load  # noqa: E402

S = "@time-count-location-fps0.2-kv{}-k0-attention_weighted-native-time-keepsurv-{}v300"
MAIN = [
    ("Token budget (random)", [("5\\%", "offline-random-keep0.05-timeline", "random5-cd-blind-a0.5-timeline"),
                               ("10\\%", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"),
                               ("25\\%", "offline-random-keep0.25-timeline", "random25-cd-blind-a0.5-timeline"),
                               ("100\\%", "timeline", "unpruned-cd-blind-a0.5-timeline")]),
    ("Selector (10\\%)", [("HERMES", "offline-hermes-keep0.1-timeline", "hermes10-cd-blind-a0.5-timeline"),
                          ("Stratified", "offline-stratified-keep0.1-timeline", "stratified10-cd-blind-a0.5-timeline"),
                          ("Random", "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline")]),
    ("HERMES streaming", [("KV 6k", S.format(6000, "timeline-"), S.format(6000, "cd-blind-a0.5-timeline-")),
                          ("KV 4k", S.format(4000, "timeline-"), S.format(4000, "cd-blind-a0.5-timeline-"))]),
]
B10 = "offline-random-keep0.1-timeline"
ABL = [
    ("Counterfactual", [("None (no video), PCD", B10, "random10-cd-blind-a0.5-timeline"),
                        ("Permuted timestamps", B10, "random10-cd-stamps-a0.5-timeline")]),
    ("Strength / cut-off", [("$\\alpha$=0.25", B10, "random10-cd-blind-a0.25-timeline"),
                            ("$\\alpha$=0.5", B10, "random10-cd-blind-a0.5-timeline"),
                            ("$\\alpha$=1.0", B10, "random10-cd-blind-a1.0-timeline"),
                            ("$\\alpha$=0.5, $\\beta$=0.2", B10, "random10-cd-blind-a0.5-b0.2-timeline")]),
    ("Prompt", [("Timeline", B10, "random10-cd-blind-a0.5-timeline"),
                ("Official", "offline-random-keep0.1", "random10-cd-blind-a0.5")]),
    ("Other methods", [("Visual-attention gain ($\\gamma$=4)", B10, "random10-vag4-timeline"),
                       ("Relevance-guided sampling", B10, "random10-rzoom-t2-m0.3-timeline"),
                       ("Self-window zoom", B10, "zoom-self-keep0.1-k32-a0.6-timeline"),
                       ("Self-window zoom + PCD", B10, "zoom-self-keep0.1-k32-a0.6-cd-blind-a0.5-timeline")]),
]
KEYS = ("acc", "miou", "r05", "gq05")


def value(rows, k):
    return 100 * sum((bool(r["_ok"]) if k == "acc" else r["_iou"] if k == "miou" else r["_iou"] >= 0.5 if k == "r05"
                      else bool(r["_ok"]) and r["_iou"] >= 0.5) for r in rows.values()) / len(rows)


def delta(m, b, k):
    qs = sorted(set(m) & set(b))
    f = lambda r: 100.0 * (bool(r["_ok"]) if k == "acc" else r["_iou"] if k == "miou" else r["_iou"] >= 0.5 if k == "r05"
                           else bool(r["_ok"]) and r["_iou"] >= 0.5)
    d = [f(m[q]) - f(b[q]) for q in qs]
    lo, hi = boot(d, n=4000)
    s = f"{sum(d) / len(d):+.1f}".replace("-", "$-$")
    return rf"\textbf{{{s}}}" if lo > 0 or hi < 0 else s


def cells(base, meth, with_leak=True):
    b = [f"{value(base, k):.1f}" for k in KEYS]
    m = [f"{value(meth, k):.1f}" for k in KEYS]
    d = [delta(meth, base, k) for k in KEYS]
    out = [f"{x} $\\rightarrow$ {y} ({z})" for x, y, z in zip(b, m, d)]
    if with_leak:
        lb, lm = leakage(base)[1], leakage(meth)[1]
        out.append(f"{lb:.0f} $\\rightarrow$ {lm:.0f}")
    return out


def table(groups, caption, label, path, leak=True):
    ncol = 2 + len(KEYS) + (1 if leak else 0)
    lines = [r"% CVPR two-column: spans both columns. Requires \usepackage{booktabs} and \usepackage{multirow}.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}", r"\toprule",
             r"& Setting & Acc. & mIoU & R@0.5 & GQ@0.5" + (r" & Early start (\%)" if leak else "") + r" \\", r"\midrule"]
    for gi, (gname, rows) in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for j, (label_, btag, mtag) in enumerate(rows):
            base, meth = load(btag), load(mtag)
            if base is None or meth is None:
                c = ["--"] * (ncol - 2)
            else:
                c = cells(base, meth, leak)
            head = (rf"\multirow{{{len(rows)}}}{{*}}{{{gname}}}" if len(rows) > 1 else gname) if j == 0 else ""
            lines.append(f"{head} & {label_} & " + " & ".join(c) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    table(MAIN, r"\textbf{Prior-contrastive decoding (PCD) on pruned video memory.} Qwen3-VL-8B on S-EMBER grounded QA "
                r"(475 questions, 300 videos), timeline-reasoning prompt. Each cell: baseline $\rightarrow$ +PCD "
                r"($\alpha$=0.5, $\beta$=0.1), with the paired difference in parentheses; bold when its 95\% paired bootstrap "
                r"interval excludes zero. Offline rows prune 64 uniform frames once to the given share of visual tokens; "
                r"streaming rows use HERMES KV compression at 0.2\,fps. Acc.: answer accuracy with the official S-EMBER judge "
                r"prompt; GQ@0.5: correct answer and IoU $\geq$ 0.5. Early start: share of intervals starting in the first 5\% "
                r"of the video (the model's temporal prior; 90\% without video).",
          "tab:pcd-main", "tables/pcd_main.tex")
    table(ABL, r"\textbf{Ablations} at 10\% of the tokens (random selector). Baseline $\rightarrow$ method (paired difference; "
               r"bold when significant). \emph{Counterfactual}: the memory contrasted against: the prompt alone (PCD) or the "
               r"same frames with permuted timestamps. \emph{Other methods}: visual-attention rebalancing during decoding, "
               r"allocation of frames and tokens by grounding-head relevance, and re-reading a self-predicted window densely.",
          "tab:pcd-ablation", "tables/pcd_ablation.tex")


if __name__ == "__main__":
    main()
