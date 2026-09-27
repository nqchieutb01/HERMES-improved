"""LaTeX table: Qwen3-VL-8B on S-EMBER (300 videos) split by question time.

Compares uniform sampling with HERMES streaming (KV 6000, k=0 and k=1 top_attention_patch) on
MCQ accuracy, grounding mIoU and grounding R@0.5, per question-time interval. Writes
tables/time_breakdown.tex. Numbers are percentages with one decimal.
"""
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = os.path.join(ROOT, "results", "qwen3_vl_8b")
METHODS = [
    ("Uniform, 32 frames", "uniform-n32-time-count-location-v300-native-time"),
    ("Uniform, 64 frames", "uniform-n64-time-count-location-v300-native-time"),
    (r"Stream 6k, $k{=}0$", "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    (r"Stream 6k, $k{=}1$", "time-count-location-fps0.2-kv6000-k1-top_attention_patch-native-time-keepsurv-v300"),
]
BINS = [(0, 80), (80, 160), (160, 320), (320, float("inf"))]


def load(task, run):
    path = os.path.join(BASE, task, run, f"{task}_scored.jsonl")
    return [json.loads(line) for line in open(path)]


def truthy(x):
    return x in (True, 1, "True", "true", "1", "1.0")


def scores(rows_by_method, metric_fns, lo, hi):
    """Per-method list of metric means (%) over questions asked in [lo, hi), plus the question count."""
    out, n = [], None
    for rows in rows_by_method:
        sel = [r for r in rows if lo <= float(r["question_time"]) < hi]
        n = len(sel)
        out.append([100 * st.mean(fn(r) for r in sel) for fn in metric_fns] if sel else [None] * len(metric_fns))
    return out, n


def main():
    mcq = [load("sember_mcq", run) for _, run in METHODS]
    grd = [load("sember_grounding", run) for _, run in METHODS]
    mcq_fns = [lambda r: truthy(r["is_correct"])]
    grd_fns = [lambda r: float(r["temporal_iou"] or 0), lambda r: truthy(r["recall_at_1_iou_0.5"])]

    groups = [(lo, hi) for lo, hi in BINS] + [(0, float("inf"))]
    lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{multirow}.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Qwen3-VL-8B on S-EMBER (300 videos) by question time, i.e.\ how much video has been "
        r"streamed when the question is asked. Streaming = HERMES at 0.2 fps (a question at $t$ seconds has "
        r"seen $t/5$ frames) with a 6k-token KV budget; $k{=}1$ keeps at least one token per frame using "
        r"\texttt{top\_attention\_patch}, the best $k{=}1$ strategy for Qwen3 at 6k in the 100-video sweep. Both streaming rows keep "
        r"timestamp tokens of surviving frames. Uniform = 32 or 64 frames sampled uniformly up to the "
        r"question, no compression. Acc = MCQ accuracy (\%); mIoU = mean temporal IoU (\%); R@0.5 = "
        r"Recall@1 at IoU $\geq$ 0.5 (\%). Best per interval in bold.}",
        r"\label{tab:sember-time-breakdown}",
        r"\begin{tabular}{llccc}", r"\toprule",
        r"& & MCQ & \multicolumn{2}{c}{Grounding} \\",
        r"\cmidrule(lr){3-3}\cmidrule(lr){4-5}",
        r"Question time & Method & Acc & mIoU & R@0.5 \\",
    ]
    for lo, hi in groups:
        m, n_mcq = scores(mcq, mcq_fns, lo, hi)
        g, n_grd = scores(grd, grd_fns, lo, hi)
        vals = [a + b for a, b in zip(m, g)]
        best = [max(v[i] for v in vals if v[i] is not None) for i in range(len(vals[0]))]
        if (lo, hi) == (0, float("inf")):
            label = "All"
            lines.append(r"\midrule[\heavyrulewidth]")
        else:
            label = f"{lo}--{hi} s" if hi != float("inf") else rf"$\geq${lo} s"
            lines.append(r"\midrule")
        head = [rf"\multirow{{{len(METHODS)}}}{{*}}{{{label}}}"]
        for j, ((name, _), v) in enumerate(zip(METHODS, vals)):
            cells = " & ".join("--" if x is None else (rf"\textbf{{{x:.1f}}}" if round(x, 1) == round(b, 1) else f"{x:.1f}")
                               for x, b in zip(v, best))
            lines.append(f"{head[0] if j == 0 else ''} & {name} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "tables", "time_breakdown.tex"), "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
