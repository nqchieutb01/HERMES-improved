"""Score our runs with TimeLens-Bench's own metric code.

Converts results.csv (our pipeline) into TimeLens' result format exactly as their evaluation/eval_dataloader.py
writes it — timestamps from their extract_time, a failed parse becomes [duration + 10, duration + 20], rounded to
the dataset unit (1.0 s) — then runs their unmodified evaluation/compute_metrics.py.
With --time-line (answers in our timeline format: "Seen: ..." lines, "Answer: ...", "Time: [s, e]"), only the text
after the last "Time:" is parsed, i.e. the model's final interval rather than the first listed moments.
Usage: python3 score.py [--time-line] <run dir with results.csv> [...]   (writes <run dir>/timelens_pred.jsonl and .log)
"""
import csv
import json
import os
import subprocess
import sys

REPO = "/home/chieu.nguyen/hermes_pool/timelens/repo"
SUBSET = "/nfs-stor/chieu.nguyen/timelens-bench/qvh_tl_subset.jsonl"  # --subset <jsonl> for other query sets
sys.path.insert(0, REPO)
from timelens.utils import extract_time  # noqa: E402


def convert(run_dir, time_line=False, subset=SUBSET):
    spans = {json.loads(l)["question_id"]: json.loads(l) for l in open(subset)}
    dumps = []
    for r in csv.DictReader(open(f"{run_dir}/results.csv")):
        a = spans[r["question_id"]]
        answer = r["pred_raw"]
        timestamps = extract_time(answer.rsplit("Time:", 1)[-1] if time_line else answer)
        if not timestamps:
            timestamps = [[a["duration"] + 10, a["duration"] + 20]]
        timestamps = [[round(s), round(e)] for s, e in timestamps]  # unit 1.0, as in eval_dataloader.py
        key = f"{a['video_id']}.mp4>>>{a['question']}>>>{a['span']}"
        dumps.append({key: {"timestamps": timestamps, "answers": answer, "duration": a["duration"]}})
    out = f"{run_dir}/timelens_pred.jsonl"
    with open(out, "w") as f:
        for d in dumps:
            f.write(json.dumps(d) + "\n")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    time_line = "--time-line" in args
    subset = args[args.index("--subset") + 1] if "--subset" in args else SUBSET
    runs = [a for i, a in enumerate(args) if a not in ("--time-line", "--subset") and (i == 0 or args[i - 1] != "--subset")]
    for run_dir in runs:
        pred = convert(run_dir.rstrip("/"), time_line, subset)
        res = subprocess.run([sys.executable, "evaluation/compute_metrics.py", "-f", pred], cwd=REPO,
                             env={**os.environ, "PYTHONPATH": REPO}, capture_output=True, text=True)
        lines = [l for l in res.stdout.splitlines() if l.startswith(("IOU", "mIOU"))]
        print(run_dir.rstrip("/").split("/")[-1], " | ".join(lines) or res.stderr[-500:])
