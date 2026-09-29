"""Distribution of gold vs predicted interval timestamps on S-EMBER grounding (Qwen3-VL-8B, uniform 64 frames).

Compares the gold evidence interval with the interval the model outputs under the official S-EMBER prompt and
under the timeline prompt, per question (475 questions). Four panels: interval start, interval end, interval
length (log scale), and start relative to the question time (0 = video start, 1 = moment the question is
asked). Curves are histograms drawn as lines (share of questions per bin, shared bins); questions whose
interval could not be parsed are left out of the predicted curves (count in the legend).
Reads logs/phase14/uniform64_by_question.json (logs/phase14/collect_uniform64.py).
Writes figures/timestamp_distribution.{png,pdf}. Run from the repo root with a Python that has matplotlib:
.venv/hermes/bin/python3 tables/plot_timestamp_distribution.py
"""
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = "logs/phase14/uniform64_by_question.json"
OUT = "figures/timestamp_distribution"
# Reference palette, categorical slots 1-3 (validated all-pairs, light mode). Line style is a second cue.
SERIES = [("gold", "Gold interval", "#2a78d6", "-", 2.6),
          ("official", "Official prompt", "#eb6834", (0, (5, 2)), 2.0),
          ("timeline", "Timeline prompt", "#1baf7a", "-", 2.0)]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def collect():
    qs = [q for v in json.load(open(SRC))["videos"] for q in v["grounding"]]
    data = {k: {"start": [], "end": [], "length": [], "rel_start": []} for k, *_ in SERIES}
    missing = {k: 0 for k, *_ in SERIES}
    for q in qs:
        spans = {"gold": q["gold_interval_s"], **{p: o["interval_s"] for p, o in q["outputs"].items()}}
        for key, span in spans.items():
            if not span or span[0] is None:
                missing[key] += 1
                continue
            s, e = sorted(span)
            data[key]["start"].append(s)
            data[key]["end"].append(e)
            data[key]["length"].append(max(e - s, 0.5))  # zero-length intervals shown in the first log bin
            data[key]["rel_start"].append(s / q["question_time_s"] if q["question_time_s"] else np.nan)
    return data, missing, len(qs)


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel("Share of questions (%)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, labelsize=8)


def panel(ax, data, field, bins, title, xlabel, log=False):
    for key, label, colour, ls, lw in SERIES:
        vals = np.asarray(data[key][field], float)
        vals = vals[np.isfinite(vals)]
        hist, edges = np.histogram(vals, bins=bins)
        share = 100 * hist / max(len(vals), 1)
        x = np.repeat(edges, 2)[1:-1]
        ax.plot(x, np.repeat(share, 2), color=colour, linestyle=ls, linewidth=lw, solid_capstyle="round")
        med = st.median(vals)
        ax.axvline(med, color=colour, linestyle=ls, linewidth=1.0, alpha=0.7)
    if log:
        ax.set_xscale("log")
    style(ax, title, xlabel)


def main():
    data, missing, n = collect()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor="#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
    panel(axes[0, 0], data, "start", np.arange(0, 630, 30), "Interval start", "Time in video (s)")
    panel(axes[0, 1], data, "end", np.arange(0, 630, 30), "Interval end", "Time in video (s)")
    panel(axes[1, 0], data, "length", np.logspace(np.log10(0.5), np.log10(640), 22), "Interval length",
          "Length (s, log scale)", log=True)
    panel(axes[1, 1], data, "rel_start", np.linspace(0, 1.2, 25), "Interval start relative to question time",
          "Start / question time (0 = video start, 1 = question asked)")
    axes[1, 1].axvline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=lw) for _, _, c, ls, lw in SERIES]
    labels = []
    for key, label, *_ in SERIES:
        med_len = st.median(data[key]["length"])
        extra = f", {missing[key]} unparsed" if missing[key] else ""
        labels.append(f"{label} (median length {med_len:.0f} s{extra})")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Gold vs predicted interval timestamps, S-EMBER grounding (Qwen3-VL-8B, uniform 64 frames, "
                 f"{n} questions)", fontsize=12, color=INK, y=0.995, fontweight="semibold")
    fig.text(0.5, 0.005, "Lines: share of questions per bin; thin vertical lines: medians.", ha="center",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93), h_pad=2.2, w_pad=2.5)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=fig.get_facecolor())
    # Table view of the same distributions (relief for the low-contrast series).
    print(f"{'series':18s} {'n':>4s} {'start med':>9s} {'end med':>8s} {'len med':>8s} {'len p25':>8s} "
          f"{'len p75':>8s} {'start/qtime med':>15s}")
    for key, label, *_ in SERIES:
        d = data[key]
        q = np.percentile(d["length"], [25, 75])
        print(f"{label:18s} {len(d['start']):4d} {st.median(d['start']):9.1f} {st.median(d['end']):8.1f} "
              f"{st.median(d['length']):8.1f} {q[0]:8.1f} {q[1]:8.1f} {np.nanmedian(d['rel_start']):15.2f}")


if __name__ == "__main__":
    main()
