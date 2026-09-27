"""LaTeX tables for the 300-video S-EMBER runs, overall numbers only.

1. tab:sember-overall-v300: MCQ accuracy, grounding mIoU and R@0.5 for uniform sampling and
   HERMES streaming at every KV budget, k=0 and k=1 (top_attention_patch, the best k=1 strategy).
   Qwen3 rows use the "surviving" timestamp mode.
2. tab:sember-qwen3-timestamps-v300: Qwen3 timestamp-token modes (default / surviving / all).

Writes tables/overall_v300.tex. Numbers are percentages with one decimal.
"""
import os

from make_v300_tables import ROOT, runs

MODELS = [("llava_ov_7b", "LLaVA-OV-7B", "none"), ("qwen3_vl_8b", "Qwen3-VL-8B", "surviving")]
METRICS = [("sember_mcq", "accuracy_percent"), ("sember_grounding", "mean_iou_percent"),
           ("sember_grounding", "recall_at_1_iou_0.5_percent")]
MODES = [("none", "default"), ("surviving", "surviving"), ("all", "all")]


def index(model):
    """(task, mode, kv, k, keep) -> overall metrics dict, skipping count-first prompt runs."""
    out = {}
    for task in ("sember_mcq", "sember_grounding"):
        for r in runs(model, task):
            if not r["countfirst"]:
                out[task, r["mode"], r["kv"], r["k"], r["keep"]] = r["metrics"]["overall"]
    return out


def fmt(v, best=None):
    if v is None:
        return "--"
    return rf"\textbf{{{v:.1f}}}" if best is not None and round(v, 1) == round(best, 1) else f"{v:.1f}"


def overall_table():
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Overall S-EMBER results on 300 videos (576 multiple-choice, 475 grounding questions). "
        r"Uniform = 32 or 64 frames sampled uniformly up to the question, no compression. Streaming = "
        r"HERMES at 0.2 fps with a KV budget in tokens; $k$ = minimum tokens kept per frame, with $k{=}1$ "
        r"using \texttt{top\_attention\_patch} (the best $k{=}1$ strategy in the 100-video sweep). "
        r"Qwen3 streaming keeps the timestamp tokens of surviving frames "
        r"(Table~\ref{tab:sember-qwen3-timestamps-v300}). Acc = MCQ accuracy (\%); mIoU = mean temporal "
        r"IoU (\%); R@0.5 = Recall@1 at IoU $\geq$ 0.5 (\%). Best per model in bold.}",
        r"\label{tab:sember-overall-v300}",
        r"\begin{tabular}{llrcccc}", r"\toprule",
        r"& & & & MCQ & \multicolumn{2}{c}{Grounding} \\",
        r"\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
        r"Model & Mode & KV / frames & $k$ & Acc & mIoU & R@0.5 \\",
    ]
    for model, label, keep in MODELS:
        idx = index(model)
        settings = [("Uniform", n, None, "none") for n in (32, 64)]
        kvs = sorted({kv for (_, mode, kv, _, _) in idx if mode == "Streaming"})
        settings += [("Streaming", kv, k, keep) for kv in kvs for k in (0, 1)
                     if any((t, "Streaming", kv, k, keep) in idx for t, _ in METRICS)]
        vals = [[(idx.get((t, *s)) or {}).get(key) for t, key in METRICS] for s in settings]
        best = [max(v[i] for v in vals if v[i] is not None) for i in range(len(METRICS))]
        lines.append(r"\midrule")
        for j, ((mode, kv, k, _), v) in enumerate(zip(settings, vals)):
            name = rf"\multirow{{{len(settings)}}}{{*}}{{{label}}}" if j == 0 else ""
            size = f"{kv} fr." if mode == "Uniform" else f"{kv}"
            if j == 2:
                lines.append(r"\cmidrule(lr){2-7}")
            cells = " & ".join(fmt(x, b) for x, b in zip(v, best))
            lines.append(f"{name} & {mode} & {size} & {'--' if k is None else k} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def timestamp_table():
    idx = index("qwen3_vl_8b")
    settings = sorted({(kv, k) for (_, mode, kv, k, _) in idx if mode == "Streaming"})
    n = len(MODES)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Qwen3-VL-8B timestamp-token modes on S-EMBER (300 videos), HERMES streaming at 0.2 fps. "
        r"Qwen3 interleaves a timestamp token group with each frame. Default = timestamp tokens are "
        r"compressed like any other token; surviving = timestamp tokens are always kept for frames that "
        r"still have visual tokens in the cache; all = timestamp tokens of every frame are always kept. "
        r"$k$ = minimum tokens kept per frame ($k{=}1$: \texttt{top\_attention\_patch}). Acc = MCQ "
        r"accuracy (\%); mIoU = mean temporal IoU (\%); R@0.5 = Recall@1 at IoU $\geq$ 0.5 (\%). "
        r"-- = not run. Best mode per row and metric in bold.}",
        r"\label{tab:sember-qwen3-timestamps-v300}",
        r"\begin{tabular}{rc" + "c" * n * len(METRICS) + "}", r"\toprule",
        r"& & \multicolumn{%d}{c}{MCQ Acc} & \multicolumn{%d}{c}{Grounding mIoU} & \multicolumn{%d}{c}{Grounding R@0.5} \\"
        % (n, n, n),
        "".join(rf"\cmidrule(lr){{{3 + i * n}-{2 + (i + 1) * n}}}" for i in range(len(METRICS))),
        r"KV & $k$ & " + " & ".join(label for _ in METRICS for _, label in MODES) + r" \\",
        r"\midrule",
    ]
    prev_kv = None
    for kv, k in settings:
        if prev_kv is not None and kv != prev_kv:
            lines.append(r"\midrule")
        cells = []
        for t, key in METRICS:
            v = [(idx.get((t, "Streaming", kv, k, mode)) or {}).get(key) for mode, _ in MODES]
            b = max((x for x in v if x is not None), default=None)
            cells += [fmt(x, b) for x in v]
        lines.append(f"{kv if kv != prev_kv else ''} & {k} & " + " & ".join(cells) + r" \\")
        prev_kv = kv
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


if __name__ == "__main__":
    tex = "\n\n".join([r"% Requires \usepackage{booktabs} and \usepackage{multirow}.",
                       overall_table(), timestamp_table()]) + "\n"
    open(os.path.join(ROOT, "tables", "overall_v300.tex"), "w").write(tex)
    print(tex)
