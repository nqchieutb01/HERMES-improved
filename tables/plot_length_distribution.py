"""Interval length (end - start) distribution: gold vs official prompt vs timeline prompt (S-EMBER grounding).

Qwen3-VL-8B, uniform 64 frames, 475 questions. Top: all questions. Bottom: one panel per question category.
Lengths are binned on a log scale; zero-length intervals (start = end) get their own "0 s" bin on the left.
Curves are histograms drawn as lines (share of the series' parsed intervals per bin); thin vertical lines
are medians. Questions whose interval could not be parsed are left out (count in the legend).
Reads logs/phase14/uniform64_by_question.json (logs/phase14/collect_uniform64.py).
Writes figures/length_distribution.{png,pdf}. Run from the repo root with a Python that has matplotlib:
.venv/hermes/bin/python3 tables/plot_length_distribution.py
"""
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = "logs/phase14/uniform64_by_question.json"
OUT = "figures/length_distribution"
SERIES = [("gold", "Gold interval", "#2a78d6", "-", 2.6),
          ("official", "Official prompt", "#eb6834", (0, (5, 2)), 2.0),
          ("timeline", "Timeline prompt", "#1baf7a", "-", 2.0)]
CATEGORIES = [("time_duration", "Duration questions"), ("counting_objects_events", "Counting questions"),
              ("location_trace", "Location questions")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
EDGES = np.logspace(0, np.log10(640), 19)  # 1 s .. 640 s; lengths in (0, 1) s fall in the first bin
ZERO_X = 0.55  # where the "0 s" bin is drawn on the log axis


def collect():
    qs = [q for v in json.load(open(SRC))["videos"] for q in v["grounding"]]
    lengths = {c: {k: [] for k, *_ in SERIES} for c in ["all"] + [c for c, _ in CATEGORIES]}
    missing = {k: 0 for k, *_ in SERIES}
    for q in qs:
        spans = {"gold": q["gold_interval_s"], **{p: o["interval_s"] for p, o in q["outputs"].items()}}
        for key, span in spans.items():
            if not span or span[0] is None:
                missing[key] += 1
                continue
            length = abs(span[1] - span[0])
            for c in ("all", q["category"]):
                lengths[c][key].append(length)
    return lengths, missing, len(qs)


def panel(ax, per_series, title, big=False):
    for key, _, colour, ls, lw in SERIES:
        vals = np.asarray(per_series[key], float)
        zero_share = 100 * np.mean(vals == 0)
        hist, _ = np.histogram(np.clip(vals[vals > 0], 1, None), bins=EDGES)
        share = 100 * hist / len(vals)
        x = np.repeat(EDGES, 2)[1:-1]
        ax.plot(x, np.repeat(share, 2), color=colour, linestyle=ls, linewidth=lw, solid_capstyle="round")
        # Zero-length bin: a short horizontal bar at the left, same style as the curve.
        ax.plot([ZERO_X * 0.85, ZERO_X * 1.15], [zero_share] * 2, color=colour, linestyle="-", linewidth=lw + 1.5,
                solid_capstyle="butt")
        ax.axvline(max(st.median(vals), ZERO_X), color=colour, linestyle=ls, linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(ZERO_X * 0.75, 700)
    ticks = [ZERO_X, 1, 3, 10, 30, 100, 300, 600]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300", "600"])
    ax.minorticks_off()
    ax.axvline(0.78, color="#b9b8b2", linewidth=0.8)  # separates the zero bin from the log axis
    ax.set_title(title, loc="left", fontsize=12 if big else 10.5, color=INK, pad=8, fontweight="semibold")
    ax.set_xlabel("Interval length, end $-$ start (s, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Share of questions (%)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    lengths, missing, n = collect()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.28)
    panel(fig.add_subplot(grid[0, :]), lengths["all"], f"All questions (n = {n})", big=True)
    for i, (cat, label) in enumerate(CATEGORIES):
        panel(fig.add_subplot(grid[1, i]), lengths[cat], f"{label} (n = {len(lengths[cat]['gold'])})")
    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=lw) for _, _, c, ls, lw in SERIES]
    labels = []
    for key, label, *_ in SERIES:
        v = lengths["all"][key]
        extra = f", {missing[key]} unparsed" if missing[key] else ""
        labels.append(f"{label} (median {st.median(v):.0f} s{extra})")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Interval length (end $-$ start): gold vs predicted, S-EMBER grounding "
                 "(Qwen3-VL-8B, uniform 64 frames)", fontsize=12.5, color=INK, y=0.99, fontweight="semibold")
    fig.text(0.5, 0.01, "Lines: share of questions per bin; the bar at 0 is the share of zero-length intervals "
             "(start = end); thin vertical lines: medians.", ha="center", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.08)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURFACE)
    # Table view of the same distributions.
    print(f"{'group':26s} {'series':16s} {'n':>4s} {'zero%':>6s} {'p10':>6s} {'p25':>6s} {'median':>6s} "
          f"{'p75':>6s} {'p90':>6s} {'mean':>6s}")
    for group, name in [("all", "All")] + CATEGORIES:
        for key, label, *_ in SERIES:
            v = np.asarray(lengths[group][key], float)
            p = np.percentile(v, [10, 25, 50, 75, 90])
            print(f"{name:26s} {label:16s} {len(v):4d} {100 * np.mean(v == 0):6.1f} "
                  + " ".join(f"{x:6.1f}" for x in p) + f" {v.mean():6.1f}")


if __name__ == "__main__":
    main()
