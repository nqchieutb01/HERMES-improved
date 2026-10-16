"""CVPR two-column LaTeX table: timeline-reasoning prompt vs the official S-EMBER grounding prompt.

Each row is one memory configuration run twice on S-EMBER grounding (475 questions, 300 videos): once with
the official prompt and once with the timeline prompt. Columns: answer accuracy (official S-EMBER judge
prompt), mIoU, R@0.5 and GQ@0.5, each as official / timeline / change. Changes are coloured by sign and bold when
the 95% paired bootstrap CI over questions excludes zero. Needs answer_judgments.jsonl for every run.
Writes tables/timeline_prompt.tex. Run from the repo root: python3 tables/make_timeline_table.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "phase13"))
from compare import boot, per_question  # noqa: E402

Q3 = "qwen3_vl_8b/sember_grounding/"
Q25 = "qwen2.5_vl_7b/sember_grounding/"
S = "time-count-location-fps0.2-kv{}-k{}-{}-native-time-keep{}{}-v300"
U = "uniform-n{}-time-count-location-{}{}v300-native-time"


def stream(kv, k, keep):
    strat = "top_attention_patch" if k else "attention_weighted"
    return lambda tl: Q3 + S.format(kv, k, strat, keep, "-timeline" if tl else "")


def uniform(model, n, prune=""):
    return lambda tl: model + U.format(n, prune + "-" if prune else "", "timeline-" if tl else "")


# (group label, [(row cells, path builder)]); row cells fill the three leading columns (two cells: the
# second spans columns 2-3).
GROUPS = [
    ("Qwen3-VL-8B, streaming (HERMES, 0.2\\,fps)", [
        (("4k", "Kept if frame survives"), stream(4000, 0, "surv")),
        (("4k", "Always kept"), stream(4000, 0, "all")),
        (("6k", "Kept if frame survives"), stream(6000, 0, "surv")),
        (("6k", "Always kept"), stream(6000, 0, "all")),
        (("10.7k", "Always kept"), stream(10700, 0, "all")),
    ]),
    ("Qwen3-VL-8B, offline (uniform frames, one pruning pass)", [
        (("32 frames", "100\\%", "None"), uniform(Q3, 32)),
        (("64 frames", "100\\%", "None"), uniform(Q3, 64)),
    ] + [
        (("64 frames", keep, label), uniform(Q3, 64, tag))
        for keep, ratio, scale in (("50\\%", "0.5", "0.7"), ("25\\%", "0.25", "0.5"), ("10\\%", "0.1", "0.35"),
                                   ("5\\%", "0.05", None))
        for label, tag in [(name, f"offline-{score}-keep{ratio}") for score, name in
                           (("hermes", "HERMES"), ("random", "Random"), ("stratified", "Stratified"))]
        + ([(f"Spatial pooling ($\\times${scale})", f"scale{scale}")] if scale else [])
    ]),
    ("Qwen2.5-VL-7B, offline (uniform frames, one pruning pass)", [
        (("64 frames", "100\\%", "None"), uniform(Q25, 64)),
        (("64 frames", "10\\%", "Stratified"), uniform(Q25, 64, "offline-stratified-keep0.1")),
    ]),
]
METRICS = [("acc", "Acc."), ("miou", "mIoU"), ("r05", "R@0.5"), ("gq05", "GQ@0.5")]
HEADS = {0: ("KV budget", "Timestamps"), 1: ("Frames", "Keep", "Pruning"), 2: ("Frames", "Keep", "Pruning")}


def ready(run):
    return all(os.path.exists(f"results/{run}/{f}")
               for f in ("sember_grounding_scored.jsonl", "answer_judgments.jsonl"))


def cells(path):
    if not (ready(path(True)) and ready(path(False))):
        return ["--"] * (3 * len(METRICS))  # run not finished or not judged yet
    t, b = per_question(path(True)), per_question(path(False))
    qs = sorted(set(t) & set(b))
    out = []
    for key, _ in METRICS:
        tv = [100 * float(t[q][key]) for q in qs]
        bv = [100 * float(b[q][key]) for q in qs]
        diffs = [x - y for x, y in zip(tv, bv)]
        d = round(sum(diffs) / len(qs), 1) + 0.0  # colour and print the rounded value (no "-0.0")
        lo, hi = boot(diffs)
        delta = f"{d:+.1f}".replace("-", "$-$")
        delta = rf"\textbf{{{delta}}}" if lo > 0 or hi < 0 else delta
        colour = "gain" if d > 0 else "loss" if d < 0 else "same"
        out += [f"{sum(bv) / len(qs):.1f}", f"{sum(tv) / len(qs):.1f}", rf"\{colour}{{{delta}}}"]
    return out


def lead_cells(lead):
    """Three leading columns; a two-item lead spans its second item over columns 2-3."""
    return list(lead) if len(lead) == 3 else [lead[0], rf"\multicolumn{{2}}{{l}}{{{lead[1]}}}"]


def main():
    lines = [
        r"% CVPR two-column: spans both columns. Requires in the preamble:",
        r"%   \usepackage{booktabs}  \usepackage{multirow}  \usepackage[table]{xcolor}",
        r"%   \definecolor{gaincol}{RGB}{0,120,60}  \definecolor{losscol}{RGB}{190,30,30}",
        r"%   \newcommand{\gain}[1]{\cellcolor{gaincol!12}\textcolor{gaincol}{#1}}",
        r"%   \newcommand{\loss}[1]{\cellcolor{losscol!12}\textcolor{losscol}{#1}}",
        r"%   \newcommand{\same}[1]{#1}",
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{\textbf{Timeline-reasoning prompt vs.\ the official S-EMBER prompt} on S-EMBER grounded "
        r"VideoQA (475 questions, 300 videos). Each row runs one memory configuration twice, changing only the "
        r"prompt. \emph{Off.}: the official benchmark prompt (answer and interval directly). \emph{TL} (timeline): the model "
        r"first lists up to 8 timestamped moments where the evidence is visible, derives the answer from them, and "
        r"gives the interval from the first to the last moment. $\Delta$ = TL $-$ Off., green for a gain "
        r"and red for a loss; bold when the 95\% paired bootstrap confidence interval over questions excludes zero. "
        r"Acc.: answer accuracy judged with the official S-EMBER judge prompt; mIoU and R@0.5: temporal grounding; "
        r"GQ@0.5: the answer is judged correct and its interval has IoU $\geq 0.5$. "
        r"\emph{Streaming}: HERMES KV compression with no per-frame token floor; \emph{Timestamps}: whether a "
        r"frame's timestamp tokens are kept while its visual tokens survive, or always. \emph{Offline}: one pruning "
        r"pass after encoding keeps the given share of visual tokens: \emph{HERMES} by attention score, "
        r"\emph{Random} at random, \emph{Stratified} an equal share of every frame; \emph{Spatial pooling} "
        r"lowers the decode resolution instead (scale $s$ keeps $\approx s^2$ of the tokens). All numbers in \%.}",
        r"\label{tab:timeline-prompt}",
        r"\begin{tabular}{lll" + "ccc" * len(METRICS) + "}",
        r"\toprule",
        r"& & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in METRICS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(METRICS))),
        r"& & & " + " & ".join(["Off.", "TL", r"$\Delta$"] * len(METRICS)) + r" \\",
    ]
    for gi, (label, rows) in enumerate(GROUPS):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{3 + 3 * len(METRICS)}}}{{l}}{{\textbf{{{label}}}}} \\")
        lines.append(" & ".join(lead_cells([rf"\textit{{{h}}}" for h in HEADS[gi]])) + " &" * (3 * len(METRICS))
                     + r" \\")
        prev = None
        for lead, path in rows:
            shown = list(lead)
            if prev and lead[0] == prev[0]:
                shown[0] = ""
                if len(lead) == 3 and lead[1] == prev[1]:
                    shown[1] = ""
                elif len(lead) == 3:
                    # Thin rule between keep-ratio blocks of the same frame count.
                    lines.append(rf"\cmidrule(lr){{2-{3 + 3 * len(METRICS)}}}")
            prev = lead
            lines.append(" & ".join(lead_cells(shown) + cells(path)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(lines) + "\n"
    open("tables/timeline_prompt.tex", "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
