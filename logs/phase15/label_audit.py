"""Phase 15: label audit (TimeLens-style) for the 475 S-EMBER grounding questions we evaluate.

Each question has three annotator answers with their own evidence interval. Measures inter-annotator
interval agreement (pairwise IoU), duration disagreement, and how scoring against one annotator (our
scorer uses answer_start/end_time) compares with the best-matching annotator, for uniform-64 runs.
Run from the repo root: python3 logs/phase15/label_audit.py
"""
import itertools
import json
import statistics as st
import sys

sys.path.insert(0, "logs/phase11")
import rule_answer as ra  # noqa: E402

B = "results/qwen3_vl_8b/sember_grounding/"


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main():
    for tag, name in (("uniform-n64-time-count-location-v300-native-time", "uni64 official"),
                      ("uniform-n64-time-count-location-timeline-v300-native-time", "uni64 timeline")):
        rows = [json.loads(l) for l in open(B + tag + "/sember_grounding_scored.jsonl")]
        pair, first_vs_best, spread, cat_pair = [], [], [], {}
        for r in rows:
            ans = json.loads(r["answers_json"])
            spans = [(float(a["start_ts"]), float(a["end_ts"])) for a in ans]
            pw = [iou(a, b) for a, b in itertools.combinations(spans, 2)]
            if not pw:
                continue
            pair.append(st.mean(pw))
            cat_pair.setdefault(r["question_category"], []).append(st.mean(pw))
            gold = (float(r["answer_start_time"]), float(r["answer_end_time"]))
            if r["interval_parseable"]:
                p = sorted((float(r["pred_start_time"]), float(r["pred_end_time"])))
                first_vs_best.append((iou(p, gold), max(iou(p, s) for s in spans), st.mean(iou(p, s) for s in spans)))
            else:
                first_vs_best.append((0.0, 0.0, 0.0))
            if r["question_category"] == "time_duration":
                d = [ra.parse_duration(a["answer_text"]) for a in ans]
                d = [x for x in d if x]
                if len(d) >= 2:
                    spread.append((max(d) - min(d)) / st.median(d))
        n = len(rows)
        if name.endswith("official"):
            print(f"inter-annotator interval IoU: mean {st.mean(pair):.2f}, median {st.median(pair):.2f}; "
                  f"questions with mean pairwise IoU < 0.5: {100 * sum(x < 0.5 for x in pair) / n:.0f}%")
            print("  by category:", {k: round(st.mean(v), 2) for k, v in cat_pair.items()})
            print(f"  duration answers: median (max-min)/median across annotators {st.median(spread):.2f}; "
                  f"annotators differ by >25%: {100 * sum(x > 0.25 for x in spread) / len(spread):.0f}%")
        f, b, m = zip(*first_vs_best)
        print(f"{name}: mIoU vs scored annotator {100 * st.mean(f):.1f} | vs best-matching annotator {100 * st.mean(b):.1f} | "
              f"vs mean over annotators {100 * st.mean(m):.1f} | R@0.5 {100 * sum(x >= .5 for x in f) / n:.1f} -> "
              f"{100 * sum(x >= .5 for x in b) / n:.1f} (best)")


if __name__ == "__main__":
    main()
