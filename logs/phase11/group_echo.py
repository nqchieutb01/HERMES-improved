"""Does the model echo the spacing of Qwen3's token groups? (no GPU)

Qwen3 merges two consecutive input frames into one token group stamped "<t seconds>" with the pair's
mean time, so the gap between neighbouring timestamps (the group spacing) depends on the sampling:
0.2 fps -> 10 s, 1 fps -> 2 s, dup:0.2 (one frame per group) -> 5 s, uniform N frames -> 2 * t / N.
If the model localises an event to one step between neighbouring timestamps, its predicted interval
length (and even the duration it states in words) should follow the group spacing across configs.

For each run: share of predicted intervals whose length is within 20% of the group spacing, the most
common predicted lengths, and for duration questions the most common stated durations and their
accuracy (official judge). For uniform sampling, questions are binned by their own group spacing.
Run from the repo root: python3 logs/phase11/group_echo.py
"""
import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_answer as ra  # noqa: E402

B = "results/qwen3_vl_8b/sember_grounding"
K = "-attention_weighted-native-time-keepsurv"
FIXED = [  # (label, run, group spacing in seconds)
    ("0.2 fps, 6k", f"time-count-location-fps0.2-kv6000-k0{K}-v300", 10.0),
    ("0.2 fps, 21.4k", f"time-count-location-fps0.2-kv21400-k0{K}-v300", 10.0),
    ("1 fps, 6k", f"time-count-location-fps1.0-kv6000-k0{K}-v300", 2.0),
    ("dup:0.2 (1 frame/group), 6k", f"time-count-location-dup0.2-kv6000-k0{K}-v300", 5.0),
    ("dup:0.2, scale 0.7, 6k", f"time-count-location-dup0.2-kv6000-k0{K}-scale0.7-v300", 5.0),
]
UNIFORM = [("uniform 32", "uniform-n32-time-count-location-v300-native-time", 32),
           ("uniform 64", "uniform-n64-time-count-location-v300-native-time", 64)]


def load(tag):
    rows = [json.loads(l) for l in open(os.path.join(B, tag, "sember_grounding_scored.jsonl"))]
    jpath = os.path.join(B, tag, "answer_judgments.jsonl")
    judge = {json.loads(l)["question_id"]: json.loads(l)["correct"] for l in open(jpath)} if os.path.exists(jpath) else {}
    return rows, judge


def lengths(rows):
    return [(r, float(r["pred_end_time"]) - float(r["pred_start_time"])) for r in rows
            if r.get("pred_start_time") not in (None, "")]


def describe(label, pairs, spacing_of, judge):
    near = [abs(L - spacing_of(r)) <= 0.2 * spacing_of(r) for r, L in pairs]
    top = collections.Counter(round(L) for _, L in pairs).most_common(3)
    dur = [(r, ra.parse_duration(ra.model_answer(r))) for r, _ in pairs if r["question_category"] == "time_duration"]
    dur = [(r, d) for r, d in dur if d is not None]
    said = collections.Counter(round(d) for _, d in dur).most_common(3)
    echo = [r for r, d in dur if abs(d - spacing_of(r)) <= 0.2 * spacing_of(r)]
    acc_echo = 100 * st.mean(bool(judge.get(r["question_id"])) for r in echo) if echo and judge else float("nan")
    sp = st.median(spacing_of(r) for r, _ in pairs)
    print(f"{label:30s} spacing {sp:5.1f}s | length ~spacing {100 * st.mean(near):4.0f}% | top lengths {top} | "
          f"stated durations {said} | durations ~spacing {len(echo)}/{len(dur)}, judged correct {acc_echo:.0f}%")


def main():
    print("length ~spacing: predicted interval length within 20% of the group spacing")
    for label, tag, spacing in FIXED:
        if not os.path.exists(os.path.join(B, tag)):
            continue
        rows, judge = load(tag)
        describe(label, lengths(rows), lambda r, s=spacing: s, judge)
    for label, tag, n in UNIFORM:
        rows, judge = load(tag)
        pairs = lengths(rows)
        spacing_of = lambda r, n=n: 2.0 * float(r["question_time"]) / n
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 1e9)):
            sub = [(r, L) for r, L in pairs if lo <= spacing_of(r) < hi]
            if len(sub) >= 20:
                describe(f"{label}, spacing {lo}-{hi:g}s", sub, spacing_of, judge)


if __name__ == "__main__":
    main()
