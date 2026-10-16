"""Token-selector comparison at matched budgets (Qwen3-VL-8B, S-EMBER, 64 frames unless noted).

Selectors: HERMES (attention + recency), uniform frame dropping (fewer uniform frames: 32/16/12/6/4 frames for
50/25/~20/~10/~5% of the tokens), random tokens, stratified (equal share per frame), spatial pooling (lower
resolution) and, as a reference, the oracle (gold-interval frames first; uses labels). Grounding with the
official and timeline prompts (Acc. = official S-EMBER judge; GQ@0.5 = correct and IoU >= 0.5) and MCQ.
Prints a text table and writes tables/selector_comparison.tex. Run from the repo root.
"""
import json
import os

G = "results/qwen3_vl_8b/sember_grounding/"
M = "results/qwen3_vl_8b/sember_mcq/"
U = "uniform-n64-time-count-location-"
LEVELS = [
    ("100\\%", [("None (64 frames)", U + "{tl}v300-native-time")]),
    ("50\\%", [("HERMES", U + "offline-hermes-keep0.5-{tl}v300-native-time"),
               ("Uniform (32 frames)", "uniform-n32-time-count-location-{tl}v300-native-time"),
               ("Random", U + "offline-random-keep0.5-{tl}v300-native-time"),
               ("Stratified", U + "offline-stratified-keep0.5-{tl}v300-native-time"),
               ("Spatial pooling ($\\times$0.7)", U + "scale0.7-{tl}v300-native-time")]),
    ("25\\%", [("HERMES", U + "offline-hermes-keep0.25-{tl}v300-native-time"),
               ("Uniform (16 frames)", "uniform-n16-time-count-location-{tl}v300-native-time"),
               ("Random", U + "offline-random-keep0.25-{tl}v300-native-time"),
               ("Stratified", U + "offline-stratified-keep0.25-{tl}v300-native-time"),
               ("Spatial pooling ($\\times$0.5)", U + "scale0.5-{tl}v300-native-time")]),
    ("10\\%", [("HERMES", U + "offline-hermes-keep0.1-{tl}v300-native-time"),
               ("Uniform (6 frames)", "uniform-n6-time-count-location-{tl}v300-native-time"),
               ("Random", U + "offline-random-keep0.1-{tl}v300-native-time"),
               ("Stratified", U + "offline-stratified-keep0.1-{tl}v300-native-time"),
               ("Spatial pooling ($\\times$0.35)", U + "scale0.35-{tl}v300-native-time"),
               ("Oracle (gold frames)", U + "offline-oracle-keep0.1-{tl}v300-native-time")]),
    ("5\\%", [("HERMES", U + "offline-hermes-keep0.05-{tl}v300-native-time"),
              ("Uniform (4 frames)", "uniform-n4-time-count-location-{tl}v300-native-time"),
              ("Random", U + "offline-random-keep0.05-{tl}v300-native-time"),
              ("Stratified", U + "offline-stratified-keep0.05-{tl}v300-native-time"),
              ("Oracle (gold frames)", U + "offline-oracle-keep0.05-{tl}v300-native-time")]),
]


def grounding(tag):
    d = G + tag
    if not (os.path.exists(f"{d}/sember_grounding_scored.jsonl") and os.path.exists(f"{d}/answer_judgments.jsonl")):
        return None
    rows = [json.loads(l) for l in open(f"{d}/sember_grounding_scored.jsonl")]
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{d}/answer_judgments.jsonl")}
    n = len(rows)
    iou = [float(r["temporal_iou"] or 0) for r in rows]
    ok = [judge[r["question_id"]] for r in rows]
    return [100 * sum(ok) / n, 100 * sum(iou) / n, 100 * sum(i >= 0.5 for i in iou) / n,
            100 * sum(a and i >= 0.5 for a, i in zip(ok, iou)) / n]


def mcq(tag):
    path = M + tag.format(tl="") + "/sember_mcq_scored.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    return 100 * sum(r["is_correct"] in (True, "True", "true", 1, "1.0") for r in rows) / len(rows)


def main():
    table = []
    for keep, items in LEVELS:
        group = []
        for name, tag in items:
            off, tl = grounding(tag.format(tl="")), grounding(tag.format(tl="timeline-"))
            group.append((name, (off or [None] * 4) + (tl or [None] * 4) + [mcq(tag)]))
        table.append((keep, group))
    f = lambda v: "--" if v is None else f"{v:.1f}"
    head = "Acc   mIoU  R@.5  GQ@.5 | Acc   mIoU  R@.5  GQ@.5 |  MCQ"
    print(f"{'Keep':6s} {'Selector':32s}   official prompt           | timeline prompt           |")
    print(f"{'':6s} {'':32s}   {head}")
    for keep, group in table:
        for i, (name, v) in enumerate(group):
            k = keep.replace("\\%", "%") if i == 0 else ""
            nm = name.replace("$\\times$", "x")
            print(f"{k:6s} {nm:32s}   " + " ".join(f"{f(x):>5s}" for x in v[:4]) + " | "
                  + " ".join(f"{f(x):>5s}" for x in v[4:8]) + f" | {f(v[8]):>5s}")
    # LaTeX (CVPR table*), best per keep ratio in bold (oracle excluded: it uses labels).
    lines = [
        r"% CVPR two-column: spans both columns. Requires \usepackage{booktabs} and \usepackage{multirow}.",
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{\textbf{Token selectors at matched budgets} with Qwen3-VL-8B on S-EMBER (300 videos). All rows "
        r"start from 64 uniformly sampled frames ($\approx$21.4k visual tokens) except \emph{Uniform}, which samples "
        r"fewer frames (32/16/12/6/4 for $\approx$50/25/19/9/6\% of the tokens). \emph{HERMES}: attention and "
        r"recency score; \emph{Random}: random tokens; \emph{Stratified}: an equal share of every frame; "
        r"\emph{Spatial pooling}: lower decode resolution (scale $s$ keeps $\approx s^2$ of the tokens); "
        r"\emph{Oracle}: gold-interval frames first (uses the labels; reference only, not bolded). Grounded QA with "
        r"the official S-EMBER prompt and with the timeline-reasoning prompt: Acc.\ is judged with the official "
        r"S-EMBER judge prompt, GQ@0.5 requires a correct answer and IoU $\geq 0.5$. MCQ: 5-way accuracy. All "
        r"numbers in \%; -- = not run. Best non-oracle per keep ratio in bold.}",
        r"\label{tab:selectors}",
        r"\begin{tabular}{llccccccccc}", r"\toprule",
        r"& & \multicolumn{4}{c}{Official prompt} & \multicolumn{4}{c}{Timeline prompt} & \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"Keep & Selector & Acc. & mIoU & R@0.5 & GQ@0.5 & Acc. & mIoU & R@0.5 & GQ@0.5 & MCQ \\",
    ]
    for keep, group in table:
        lines.append(r"\midrule")
        cand = [v for name, v in group if not name.startswith("Oracle")]
        best = [max((v[i] for v in cand if v[i] is not None), default=None) for i in range(9)]
        for j, (name, v) in enumerate(group):
            cells = []
            for i, x in enumerate(v):
                s = f(x)
                if x is not None and len(cand) > 1 and not name.startswith("Oracle") and round(x, 1) == round(best[i], 1):
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            label = name if not name.startswith("Oracle") else rf"\textit{{{name}}}"
            head_cell = (rf"\multirow{{{len(group)}}}{{*}}{{{keep}}}" if len(group) > 1 else keep) if j == 0 else ""
            lines.append(f"{head_cell} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open("tables/selector_comparison.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
