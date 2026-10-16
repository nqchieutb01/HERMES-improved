"""CVPR two-column LaTeX table: per-task performance (duration, counting, location) of the reference inputs,
the token selectors at 50% of the tokens and the evidence-window oracle, with both grounding prompts.

Qwen3-VL-8B on S-EMBER grounded QA (475 questions, 300 videos: 191 duration, 184 counting, 100 location).
Per task: Acc. (official S-EMBER judge prompt), mIoU, R@0.5 and GQ@0.5 (correct answer and IoU >= 0.5).
Writes tables/task_breakdown.tex. Run from the repo root: python3 tables/make_task_table.py
"""
import json
import os

G = "results/qwen3_vl_8b/sember_grounding/"
U = "uniform-n64-time-count-location-"
TASKS = [("time_duration", "Duration"), ("counting_objects_events", "Counting"), ("location_trace", "Location")]
GROUPS = [
    ("Reference", [("No video (blind)", U + "dx-blind-{tl}v300-native-time"),
                   ("Uniform, 64 frames (all tokens)", U + "{tl}v300-native-time")]),
    ("50\\%", [("Uniform, 32 frames", "uniform-n32-time-count-location-{tl}v300-native-time"),
               ("HERMES", U + "offline-hermes-keep0.5-{tl}v300-native-time"),
               ("Random", U + "offline-random-keep0.5-{tl}v300-native-time"),
               ("Stratified", U + "offline-stratified-keep0.5-{tl}v300-native-time"),
               ("Spatial pooling ($\\times$0.7)", U + "scale0.7-{tl}v300-native-time")]),
    ("Upper bound", [("\\textit{Evidence-window oracle}", U + "oracle-window-{tl}v300-native-time")]),
]


def per_task(tag):
    d = G + tag
    if not (os.path.exists(f"{d}/sember_grounding_scored.jsonl") and os.path.exists(f"{d}/answer_judgments.jsonl")):
        return [None] * 12
    rows = [json.loads(l) for l in open(f"{d}/sember_grounding_scored.jsonl")]
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{d}/answer_judgments.jsonl")}
    out = []
    for cat, _ in TASKS:
        sub = [r for r in rows if r["question_category"] == cat]
        n = len(sub)
        iou = [float(r["temporal_iou"] or 0) for r in sub]
        ok = [judge[r["question_id"]] for r in sub]
        out += [100 * sum(ok) / n, 100 * sum(iou) / n, 100 * sum(i >= 0.5 for i in iou) / n,
                100 * sum(a and i >= 0.5 for a, i in zip(ok, iou)) / n]
    return out


def counts():
    rows = [json.loads(l) for l in open(G + U + "v300-native-time/sember_grounding_scored.jsonl")]
    return {cat: sum(r["question_category"] == cat for r in rows) for cat, _ in TASKS}


def main():
    f = lambda v: "--" if v is None else f"{v:.1f}"
    n = counts()
    lines = [
        r"% CVPR two-column: spans both columns. Requires \usepackage{booktabs} and \usepackage{multirow}.",
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{\textbf{Per-task results} for Qwen3-VL-8B on S-EMBER grounded QA (300 videos): time duration "
        rf"({n['time_duration']} questions), counting ({n['counting_objects_events']}) and location trace "
        rf"({n['location_trace']}). Inputs as in Table~\ref{{tab:oracle}}: the question alone, 64 uniform frames with all "
        r"tokens, the token selectors at 50\% of the visual tokens (uniform 32 frames; HERMES, random and stratified "
        r"pruning of 64 frames; spatial pooling at scale 0.7), and the evidence-window oracle (64 frames inside the gold "
        r"interval, timestamp text elsewhere). Acc.: answer accuracy with the official S-EMBER judge prompt; mIoU and "
        r"R@0.5: temporal grounding; GQ@0.5: correct answer and IoU $\geq 0.5$. Top: official S-EMBER prompt; bottom: "
        r"timeline-reasoning prompt. All numbers in \%. Best selector at 50\% in bold, per prompt.}",
        r"\label{tab:tasks}",
        r"\begin{tabular}{ll" + "cccc" * len(TASKS) + "}", r"\toprule",
        r"& & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{name}}}" for _, name in TASKS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(TASKS))),
        r"Tokens & Input & " + " & ".join(["Acc.", "mIoU", "R@0.5", "GQ@0.5"] * len(TASKS)) + r" \\",
    ]
    for prompt, tl in (("Official prompt", ""), ("Timeline prompt", "timeline-")):
        lines += [r"\midrule", rf"\multicolumn{{{2 + 4 * len(TASKS)}}}{{l}}{{\textbf{{{prompt}}}}} \\"]
        print(f"\n{prompt}")
        for gi, (label, rows) in enumerate(GROUPS):
            if gi:
                lines.append(rf"\cmidrule(lr){{1-{2 + 4 * len(TASKS)}}}")
            vals = {name: per_task(tag.format(tl=tl)) for name, tag in rows}
            compare = label == "50\\%"
            best = [max((v[i] for v in vals.values() if v[i] is not None), default=None) for i in range(12)]
            for j, (name, _) in enumerate(rows):
                v = vals[name]
                cells = [f(x) for x in v]
                if compare:
                    cells = [rf"\textbf{{{c}}}" if x is not None and round(x, 1) == round(best[i], 1) else c
                             for i, (c, x) in enumerate(zip(cells, v))]
                head = (rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if len(rows) > 1 else label) if j == 0 else ""
                lines.append(f"{head} & {name} & " + " & ".join(cells) + r" \\")
                plain = name.replace("\\textit{", "").replace("}", "").replace("($\\times$0.7)", "x0.7")
                print(f"  {(label.replace(chr(92), '') if j == 0 else ''):12s} {plain:32s} "
                      + " | ".join(" ".join(f"{f(x):>5s}" for x in v[4 * t:4 * t + 4]) for t in range(len(TASKS))))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open("tables/task_breakdown.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
