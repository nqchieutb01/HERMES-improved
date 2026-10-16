"""Zoom windows for coverage-then-zoom (prune_score=zoom).

  self  : from a coarse-pass run's own output (no labels): the span of its "Seen" moments and its interval,
          widened by max(min_margin, margin x span), clipped to [0, question time]; questions with no parseable
          time get the whole video.
  gold  : the gold evidence interval (localisation upper bound; uses labels).
  full  : the whole video [0, question time] (control: extra frames without localisation).
Usage: python3 logs/phase17/make_windows.py <mode> <out.json> [coarse_run_dir] [--margin 1.0] [--min-margin 30]
"""
import argparse
import json
import re

SEEN = re.compile(r"^\s*Seen:\s*([\d.]+)", re.M)
REF = "results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-v300-native-time/sember_grounding_scored.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("self", "gold", "full"))
    ap.add_argument("out")
    ap.add_argument("run", nargs="?")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--min-margin", type=float, default=30.0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.run}/sember_grounding_scored.jsonl" if a.mode == "self" else REF)]
    out = {}
    for r in rows:
        qt = float(r["question_time"])
        if a.mode == "gold":
            out[r["question_id"]] = [[float(r["answer_start_time"]), min(float(r["answer_end_time"]), qt)]]
            continue
        if a.mode == "full":
            out[r["question_id"]] = [[0.0, qt]]
            continue
        times = [float(x) for x in SEEN.findall(r["pred_raw"] or "")]
        if r["interval_parseable"]:
            times += [float(r["pred_start_time"]), float(r["pred_end_time"])]
        times = [t for t in times if 0 <= t <= qt]
        if not times:
            out[r["question_id"]] = [[0.0, qt]]
            continue
        s, e = min(times), max(times)
        m = max(a.min_margin, a.margin * (e - s))
        out[r["question_id"]] = [[max(0.0, s - m), min(qt, e + m)]]
    json.dump(out, open(a.out, "w"), indent=0)
    lens = sorted((w[0][1] - w[0][0]) for w in out.values())
    print(f"{a.mode}: {len(out)} windows, median length {lens[len(lens) // 2]:.0f} s")


if __name__ == "__main__":
    main()
