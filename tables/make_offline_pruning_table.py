"""CVPR two-column LaTeX table: offline token pruning with Qwen3-VL-8B on S-EMBER (300 videos).

Rows: keep ratio x pruning score (HERMES / random / stratified / recent) plus spatial pooling at roughly
matched token count. Columns: open-ended (answer accuracy with the official S-EMBER judge prompt, mIoU,
R@0.5, GQ@0.3, GQ@0.5) and MCQ last. Needs answer_judgments.jsonl from logs/phase11/judge_grounding.py.
Writes tables/offline_pruning.tex. Run from the repo root: python3 tables/make_offline_pruning_table.py
"""
import json
import os

B = "results/qwen3_vl_8b"
U = "uniform-n64-time-count-location-{}-v300-native-time"
KEEPS = [("0.5", "0.7"), ("0.25", "0.5"), ("0.2", "0.45"), ("0.1", "0.35"), ("0.05", None)]
SCORES = [("hermes", "HERMES"), ("random", "Random"), ("stratified", "Stratified"), ("recent", "Recent")]


def truthy(v):
    return v in (True, "True", "true", 1, "1.0")


def metrics(tag):
    g = f"{B}/sember_grounding/{tag}"
    rows = [json.loads(l) for l in open(f"{g}/sember_grounding_scored.jsonl")]
    judge = {json.loads(l)["question_id"]: json.loads(l)["correct"] for l in open(f"{g}/answer_judgments.jsonl")}
    mcq = [json.loads(l) for l in open(f"{B}/sember_mcq/{tag}/sember_mcq_scored.jsonl")]
    n = len(rows)
    iou = [float(r["temporal_iou"] or 0) for r in rows]
    ok = [bool(judge[r["question_id"]]) for r in rows]
    return [100 * sum(ok) / n, 100 * sum(iou) / n, 100 * sum(truthy(r["recall_at_1_iou_0.5"]) for r in rows) / n,
            100 * sum(a and i >= 0.3 for a, i in zip(ok, iou)) / n,
            100 * sum(a and i >= 0.5 for a, i in zip(ok, iou)) / n,
            100 * sum(truthy(r["is_correct"]) for r in mcq) / len(mcq)]


def complete(tag):
    return (os.path.exists(f"{B}/sember_grounding/{tag}/answer_judgments.jsonl")
            and os.path.exists(f"{B}/sember_mcq/{tag}/sember_mcq_scored.jsonl"))


def main():
    groups = [("100\\%", [("None", "uniform-n64-time-count-location-v300-native-time")])]
    for keep, pool in KEEPS:
        items = [(label, U.format(f"offline-{s}-keep{keep}")) for s, label in SCORES
                 if complete(U.format(f"offline-{s}-keep{keep}"))]
        if pool and complete(U.format(f"scale{pool}")):
            items.append((f"Spatial pooling ($\\times${pool})", U.format(f"scale{pool}")))
        groups.append((f"{float(keep) * 100:g}\\%", items))
    lines = [
        r"% CVPR two-column: spans both columns. Requires \usepackage{booktabs} and \usepackage{multirow}.",
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{8pt}",
        r"\caption{\textbf{Offline token pruning} with Qwen3-VL-8B on S-EMBER (300 videos). After encoding 64 "
        r"uniformly sampled frames ($\approx$21.4k visual tokens), one pruning pass keeps the given fraction of "
        r"visual tokens. \emph{HERMES}: highest attention score; \emph{Random}: random tokens; \emph{Stratified}: "
        r"an equal share of every frame, choosing each frame's highest-scoring tokens; \emph{Recent}: most recent "
        r"tokens; \emph{Spatial pooling}: lower decode resolution at roughly matched token count (scale $s$ keeps "
        r"$\approx s^2$ of the tokens, e.g.\ $\times$0.35 keeps $\approx$12\%). \emph{Open-ended}: Acc.\ is answer "
        r"accuracy judged with the official S-EMBER judge prompt; mIoU and R@0.5 measure temporal grounding; "
        r"GQ@$\tau$ counts an answer only if it is judged correct and its interval has IoU $\geq \tau$. MCQ: 5-way "
        r"multiple-choice accuracy. All numbers in \%. Best per keep ratio in bold.}",
        r"\label{tab:offline-pruning}",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"& & \multicolumn{5}{c}{Open-ended} & \\", r"\cmidrule(lr){3-7}",
        r"Keep & Pruning & Acc. & mIoU & R@0.5 & GQ@0.3 & GQ@0.5 & MCQ \\", r"\midrule",
    ]
    for gi, (keep, items) in enumerate(groups):
        vals = [metrics(tag) for _, tag in items]
        best = [max(v[i] for v in vals) for i in range(6)]
        if gi:
            lines.append(r"\midrule")
        for k, ((label, _), v) in enumerate(zip(items, vals)):
            cells = [rf"\textbf{{{x:.1f}}}" if len(items) > 1 and round(x, 1) == round(b, 1) else f"{x:.1f}"
                     for x, b in zip(v, best)]
            head = (rf"\multirow{{{len(items)}}}{{*}}{{{keep}}}" if len(items) > 1 else keep) if k == 0 else ""
            lines.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(lines) + "\n"
    open("tables/offline_pruning.tex", "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
