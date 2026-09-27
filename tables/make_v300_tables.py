"""Build LaTeX tables for all 300-video S-EMBER runs (MCQ and grounding).

Reads results/<model>/<task>/*-v300*/..._metrics.json and writes
tables/v300_results.tex. Numbers are percentages with one decimal.
"""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = [("llava_ov_7b", "LLaVA-OV-7B"), ("qwen3_vl_8b", "Qwen3-VL-8B")]
CATS = [("time_duration", "Time"), ("counting_objects_events", "Count"), ("location_trace", "Location")]
KEEP_LABEL = {"none": "default", "surviving": "surviving", "all": "all"}
STREAM = re.compile(
    r"time-count-location-fps(?P<fps>[\d.]+)-kv(?P<kv>\d+)-k(?P<k>\d)-(?P<strat>[a-z_]+?)"
    r"(?P<native>-native-time)?(?P<keep>-keepsurv|-keepall)?(?P<countfirst>-countfirst)?-v300$")
UNIFORM = re.compile(r"uniform-n(?P<n>\d+)-time-count-location-v300(?P<native>-native-time)?$")


def runs(model, task):
    base = os.path.join(ROOT, "results", model, task)
    metrics_name = f"{task}_metrics.json"
    out = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, metrics_name)
        if "v300" not in name or not os.path.exists(path):
            continue
        m = STREAM.match(name) or UNIFORM.match(name)
        if not m:
            continue
        g = m.groupdict()
        info = {"metrics": json.load(open(path)), "countfirst": bool(g.get("countfirst")),
                "native": bool(g.get("native"))}
        if "n" in g:
            info.update(mode="Uniform", kv=int(g["n"]), k=None, keep="none", sort=(1, int(g["n"]), 0, 0))
        else:
            keep = {"-keepsurv": "surviving", "-keepall": "all"}.get(g["keep"], "none")
            info.update(mode="Streaming", kv=int(g["kv"]), k=int(g["k"]), keep=keep,
                        sort=(0, int(g["kv"]), int(g["k"]), ["none", "surviving", "all"].index(keep)))
        out.append(info)
    return sorted(out, key=lambda r: r["sort"])


def setting_cells(r):
    if r["mode"] == "Uniform":
        return f"Uniform & {r['kv']} fr. & -- & --"
    keep = KEEP_LABEL[r["keep"]] if r["native"] else "--"
    return f"Streaming & {r['kv']} & {r['k']} & {keep}"


def mcq_table():
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{S-EMBER multiple-choice accuracy (\%) on 300 videos (576 questions; 5 options, chance = 20\%). "
        r"Streaming = HERMES at 0.2 fps with KV budget in tokens; $k$ = minimum tokens kept per frame "
        r"($k{=}1$ uses \texttt{top\_attention\_patch}); Timestamps = which Qwen3 timestamp tokens survive compression (default = compressed like other tokens; "
        r"surviving = kept for frames with surviving tokens; all = always kept). "
        r"Uniform = 32 or 64 frames sampled uniformly up to the question, no compression.}",
        r"\label{tab:sember-mcq-v300}",
        r"\begin{tabular}{llrclcccc}", r"\toprule",
        r"Model & Mode & KV / frames & $k$ & Timestamps & Overall & Time & Count & Location \\", r"\midrule",
    ]
    for i, (model, label) in enumerate(MODELS):
        rows = [r for r in runs(model, "sember_mcq") if not r["countfirst"]]
        if i:
            lines.append(r"\midrule")
        for j, r in enumerate(rows):
            o, pc = r["metrics"]["overall"], r["metrics"]["per_category"]
            name = rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if j == 0 else ""
            cats = " & ".join(f"{pc[c]['accuracy_percent']:.1f}" for c, _ in CATS)
            lines.append(f"{name} & {setting_cells(r)} & {o['accuracy_percent']:.1f} & {cats} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def grounding_table():
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{S-EMBER grounding on 300 videos (475 questions): mean temporal IoU (mIoU, \%) and "
        r"Recall@1 at IoU 0.5 (R@0.5, \%), overall and mIoU per question type. Settings as in "
        r"Table~\ref{tab:sember-mcq-v300}; answers up to 128 tokens.}",
        r"\label{tab:sember-grounding-v300}",
        r"\begin{tabular}{llrclcccccc}", r"\toprule",
        r"& & & & & \multicolumn{2}{c}{Overall} & \multicolumn{3}{c}{mIoU by type} \\",
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
        r"Model & Mode & KV / frames & $k$ & Timestamps & mIoU & R@0.5 & Time & Count & Location \\", r"\midrule",
    ]
    for i, (model, label) in enumerate(MODELS):
        rows = runs(model, "sember_grounding")
        if i:
            lines.append(r"\midrule")
        for j, r in enumerate(rows):
            o, pc = r["metrics"]["overall"], r["metrics"]["per_category"]
            name = rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if j == 0 else ""
            cats = " & ".join(f"{pc[c]['mean_iou_percent']:.1f}" for c, _ in CATS)
            lines.append(f"{name} & {setting_cells(r)} & {o['mean_iou_percent']:.1f} & "
                         f"{o['recall_at_1_iou_0.5_percent']:.1f} & {cats} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def counting_prompt_table():
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Counting prompt on S-EMBER multiple choice (300 videos; streaming, KV 6000, $k{=}0$). "
        r"Official = answer with a letter only; Count-first = state the count, then the letter. "
        r"Only the 201 counting questions change prompt.}",
        r"\label{tab:sember-count-prompt-v300}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"Model & Prompt & Counting & Overall \\", r"\midrule",
    ]
    for i, (model, label) in enumerate(MODELS):
        if i:
            lines.append(r"\midrule")
        rows = [r for r in runs(model, "sember_mcq")
                if r["mode"] == "Streaming" and r["kv"] == 6000 and r["k"] == 0 and r["keep"] == "none"]
        for j, r in enumerate(sorted(rows, key=lambda r: r["countfirst"])):
            o, pc = r["metrics"]["overall"], r["metrics"]["per_category"]
            name = rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(f"{name} & {'Count-first' if r['countfirst'] else 'Official'} & "
                         f"{pc['counting_objects_events']['accuracy_percent']:.1f} & {o['accuracy_percent']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


K1_STRATEGIES = [
    ("top_patch", r"Top patch (HERMES score)"),
    ("top_attention_patch", r"Top patch (attention)"),
    ("mean", r"Mean pool"),
    ("attention_weighted", r"Attention-weighted pool"),
    ("softmax_score_weighted", r"Softmax-score pool ($T{=}0.1$)"),
]
# (task, metric key) columns for LLaVA-OV-7B; k=0 dirs predate the strategy suffix in grounding.
K1_METRICS = [
    ("sember_mcq", "accuracy_percent"),
    ("sember_grounding", "mean_iou_percent"),
    ("sember_grounding", "recall_at_1_iou_0.5_percent"),
]
K1_K0_DIR = {"sember_mcq": "{base}-k0-attention_weighted", "sember_grounding": "{base}-k0"}
KVS = [4000, 6000]


def _v100_metric(task, name, key):
    path = os.path.join(ROOT, "results", "llava_ov_7b", task, name, f"{task}_metrics.json")
    return json.load(open(path))["overall"][key] if os.path.exists(path) else None


def k1_strategy_table():
    """LLaVA-OV-7B frame-summary strategies for k=1 on the 100-video subset (the only split they were swept on)."""
    def fmt(v, bold=False):
        if v is None:
            return "--"
        return rf"\textbf{{{v:.1f}}}" if bold else f"{v:.1f}"

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Frame-summary strategies for the per-frame floor $k{=}1$ with LLaVA-OV-7B on the "
        r"100-video S-EMBER subset (183 multiple-choice, 159 grounding questions). Streaming = HERMES at "
        r"0.2 fps with a KV budget in tokens. When compression would drop every token of a frame, "
        r"the \emph{patch} strategies keep that frame's single highest-scoring token (by HERMES retention "
        r"score or by raw attention); the \emph{pool} strategies instead insert one synthetic K/V token "
        r"pooled over the frame (uniform mean, attention-weighted, or softmax over retention scores). "
        r"Uniform = 32 or 64 frames sampled uniformly up to the question, no compression. "
        r"Acc = MCQ accuracy (\%); mIoU = mean temporal IoU (\%); R@0.5 = Recall@1 at IoU $\geq$ 0.5 (\%). "
        r"Best $k{=}1$ strategy per KV budget in bold.}",
        r"\label{tab:sember-k1-strategies}",
        r"\begin{tabular}{lrclccc}", r"\toprule",
        r"& & & & MCQ & \multicolumn{2}{c}{Grounding} \\",
        r"\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
        r"Mode & KV / frames & $k$ & Strategy & Acc & mIoU & R@0.5 \\", r"\midrule",
    ]
    for n in (32, 64):
        cells = " & ".join(fmt(_v100_metric(t, f"uniform-n{n}-time-count-location-v100", k)) for t, k in K1_METRICS)
        lines.append(f"Uniform & {n} fr. & -- & -- & {cells} \\\\")
    for kv in KVS:
        base = f"time-count-location-fps0.2-kv{kv}"
        k1 = {strat: [_v100_metric(t, f"{base}-k1-{strat}", k) for t, k in K1_METRICS] for strat, _ in K1_STRATEGIES}
        best = [max(v[i] for v in k1.values() if v[i] is not None) for i in range(len(K1_METRICS))]
        lines.append(r"\midrule")
        k0 = " & ".join(fmt(_v100_metric(t, K1_K0_DIR[t].format(base=base), k)) for t, k in K1_METRICS)
        lines.append(f"Streaming & {kv} & 0 & -- & {k0} \\\\")
        for strat, label in K1_STRATEGIES:
            cells = " & ".join(fmt(v, v == b) for v, b in zip(k1[strat], best))
            lines.append(f"Streaming & {kv} & 1 & {label} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    tex = "\n\n".join([
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}.",
        mcq_table(), grounding_table(), counting_prompt_table(), k1_strategy_table()]) + "\n"
    out = os.path.join(ROOT, "tables", "v300_results.tex")
    open(out, "w").write(tex)
    print(tex)
