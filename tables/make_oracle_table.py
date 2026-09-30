"""CVPR two-column LaTeX table: blind, uniform frame dropping, HERMES, random, stratified, spatial pooling and oracles.

Qwen3-VL-8B on S-EMBER grounded QA (475 questions, 300 videos). Rows: no video (blind); 64 uniform frames
(all tokens); at 50/25/10/5% of the tokens: uniform frame dropping (32/16/6/4 frames), HERMES offline pruning
of 64 frames, random, stratified, spatial pooling, and the oracle (gold-interval frames first); the evidence-window oracle (64 frames inside the
gold interval, timestamp text elsewhere). Columns: share of kept visual tokens inside the gold interval, then
Acc., mIoU, R@0.5 and GQ@0.5 with the official and the timeline prompt.
Writes tables/oracle_comparison.tex. Run from the repo root: python3 tables/make_oracle_table.py
"""
import sys

sys.path.insert(0, "logs/phase16")
sys.path.insert(0, "tables")
from evidence_share import from_retention, uniform_frames  # noqa: E402
from selector_comparison import grounding  # noqa: E402

U = "uniform-n64-time-count-location-"


def uni(n):
    return f"uniform-n{n}-time-count-location-{{tl}}v300-native-time"


def share_uniform(n):
    return 100 * uniform_frames(n, None)[1]


def share_ret(tag):
    v = from_retention(tag)
    return 100 * v[1] if v else None


def share_even():
    """Selectors that keep (in expectation) the same share of every frame: the evidence share equals the
    unpruned one. Shown with a dagger."""
    return ("dagger", share_uniform(64))


def extra(keep, pool):
    ratio = {"50": "0.5", "25": "0.25", "10": "0.1", "5": "0.05"}[keep]
    rows = [("Random", U + f"offline-random-keep{ratio}-{{tl}}v300-native-time",
             (lambda: share_ret("dx-ret-random-keep0.1")) if keep == "10" else share_even),
            ("Stratified", U + f"offline-stratified-keep{ratio}-{{tl}}v300-native-time",
             (lambda: share_ret("dx-ret-stratified-keep0.1")) if keep == "10" else share_even)]
    if pool:
        rows.append((f"Spatial pooling ($\\times${pool})", U + f"scale{pool}-{{tl}}v300-native-time", share_even))
    return rows


GROUPS = [
    ("Reference", [
        ("No video (blind)", U + "dx-blind-{tl}v300-native-time", None),
        ("Uniform, 64 frames (all tokens)", U + "{tl}v300-native-time", lambda: share_uniform(64)),
    ]),
    ("50\\%", [
        ("Uniform, 32 frames", uni(32), lambda: share_uniform(32)),
        ("HERMES", U + "offline-hermes-keep0.5-{tl}v300-native-time", lambda: share_ret("dx-ret-hermes-keep0.5")),
    ] + extra("50", "0.7")),
    ("25\\%", [
        ("Uniform, 16 frames", uni(16), lambda: share_uniform(16)),
        ("HERMES", U + "offline-hermes-keep0.25-{tl}v300-native-time", lambda: share_ret("dx-ret-hermes-keep0.25")),
    ] + extra("25", "0.5")),
    ("10\\%", [
        ("Uniform, 6 frames", uni(6), lambda: share_uniform(6)),
        ("HERMES", U + "offline-hermes-keep0.1-{tl}v300-native-time", lambda: share_ret("dx-ret-hermes-keep0.1")),
    ] + extra("10", "0.35") + [
        ("Oracle (gold frames first)", U + "offline-oracle-keep0.1-{tl}v300-native-time",
         lambda: share_ret("offline-oracle-keep0.1")),
    ]),
    ("5\\%", [
        ("Uniform, 4 frames", uni(4), lambda: share_uniform(4)),
        ("HERMES", U + "offline-hermes-keep0.05-{tl}v300-native-time", lambda: share_ret("dx-ret-hermes-keep0.05")),
    ] + extra("5", None) + [
        ("Oracle (gold frames first)", U + "offline-oracle-keep0.05-{tl}v300-native-time",
         lambda: share_ret("offline-oracle-keep0.05")),
    ]),
    ("Upper bound", [
        ("Evidence-window oracle", U + "oracle-window-{tl}v300-native-time", lambda: 100.0),
    ]),
]


def main():
    f = lambda v: "--" if v is None else f"{v:.1f}"
    lines = [
        r"% CVPR two-column: spans both columns. Requires \usepackage{booktabs} and \usepackage{multirow}.",
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{\textbf{Token selectors against blind and oracle inputs} for Qwen3-VL-8B on S-EMBER grounded QA "
        r"(475 questions, 300 videos). \emph{No video}: the question alone. \emph{Uniform}: $N$ frames sampled "
        r"uniformly up to the question time, all tokens kept (64 frames $\approx$21.4k visual tokens; 32/16/6/4 frames "
        r"match 50/25/9/6\% of them). \emph{HERMES}, \emph{Random}, \emph{Stratified}: 64 frames pruned once to the "
        r"given share of visual tokens by the HERMES attention-and-recency score, at random, or with an equal share of "
        r"every frame; \emph{Spatial pooling}: 64 frames decoded at lower resolution (scale $s$ keeps $\approx s^2$ "
        r"of the tokens). \emph{Oracle}: the same pruning, but tokens of frames inside the "
        r"gold evidence interval are kept first (uses the labels). \emph{Evidence-window oracle}: 64 frames sampled "
        r"inside the gold interval only, with timestamp text every 5\,s elsewhere and no pruning. \emph{Gold}: share "
        r"of the kept visual tokens that lie inside the gold interval ($^\dagger$: by construction or in expectation "
        r"equal to the unpruned share, as every frame keeps the same share of its tokens). Acc.: answer accuracy with the official S-EMBER "
        r"judge prompt; mIoU and R@0.5: temporal grounding; GQ@0.5: correct answer and IoU $\geq 0.5$. Official: the "
        r"S-EMBER prompt; Timeline: the model lists timestamped moments before answering. All numbers in \%.}",
        r"\label{tab:oracle}",
        r"\begin{tabular}{llccccccccc}", r"\toprule",
        r"& & & \multicolumn{4}{c}{Official prompt} & \multicolumn{4}{c}{Timeline prompt} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
        r"Tokens & Input & Gold & Acc. & mIoU & R@0.5 & GQ@0.5 & Acc. & mIoU & R@0.5 & GQ@0.5 \\",
    ]
    print(f"{'Tokens':12s} {'Input':34s} {'Gold':>5s} | official Acc mIoU R@.5 GQ@.5 | timeline Acc mIoU R@.5 GQ@.5")
    for gi, (label, rows) in enumerate(GROUPS):
        lines.append(r"\midrule")
        for j, (name, tag, share) in enumerate(rows):
            off, tl = grounding(tag.format(tl="")), grounding(tag.format(tl="timeline-"))
            sh = share() if share else None
            dagger = isinstance(sh, tuple)
            sh = sh[1] if dagger else sh
            vals = [sh] + (off or [None] * 4) + (tl or [None] * 4)
            head = (rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if len(rows) > 1 else label) if j == 0 else ""
            cells = [f(v) for v in vals]
            if dagger:
                cells[0] += r"$^\dagger$"
            if name.startswith(("Oracle", "Evidence")):
                name = rf"\textit{{{name}}}"
            lines.append(f"{head} & {name} & " + " & ".join(cells) + r" \\")
            print(f"{(label if j == 0 else ''):12s} {name.replace(chr(92) + 'textit{', '').rstrip('}'):34s} {cells[0]:>5s} | "
                  + " ".join(f"{c:>5s}" for c in cells[1:5]) + " | " + " ".join(f"{c:>5s}" for c in cells[5:]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open("tables/oracle_comparison.tex", "w").write("\n".join(lines).replace("\\%\\%", "\\%") + "\n")


if __name__ == "__main__":
    main()
