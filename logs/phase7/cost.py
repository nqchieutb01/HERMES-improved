"""Compute cost of S-EMBER runs from inference logs (Qwen3-VL-8B, 300 videos, no GPU needed).

Per run (all chunks summed): GPU wall time after model load, frames encoded (streamed or, for
uniform, re-encoded per question), questions answered, and mean time-to-first-token.
Run from the repo root: python3 logs/phase7/cost.py
"""
import glob
import os
import re
import statistics as st
from datetime import datetime

B = "results/qwen3_vl_8b"
TAGS = [
    ("uniform 32", "uniform-n32-time-count-location-v300-native-time"),
    ("uniform 64", "uniform-n64-time-count-location-v300-native-time"),
    ("fps0.2 kv6000", "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("fps0.2 kv10700", "time-count-location-fps0.2-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
    ("fps1.0 kv6000", "time-count-location-fps1.0-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("s1x60 kv6000", "time-count-location-s1x60-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("s1x160 kv10700", "time-count-location-s1x160-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid16x16 kv6000", "time-count-location-grid16x16-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid32x8 kv6000", "time-count-location-grid32x8-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid32x8 kv10700", "time-count-location-grid32x8-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
]
STAMP = re.compile(r"^\[I (\d{6} \d\d:\d\d:\d\d) ")
ENC = re.compile(r"Encoding (?:uniform )?frames (\d+) to (\d+)")


def log_cost(path):
    start = end = None
    frames = answers = 0
    ttft = []
    for line in open(path, errors="replace"):
        m = STAMP.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%y%m%d %H:%M:%S")
            if "Effective inference settings" in line:
                start = t
            end = t
        m = ENC.search(line)
        if m:
            frames += int(m.group(2)) - int(m.group(1)) + 1
        if line.startswith("Pred Answer:"):
            answers += 1
        if line.startswith("TTFT:"):
            ttft.append(float(line.split()[1]))
    # Some runs stop timestamping after setup; the log's mtime marks when the worker finished.
    end = max(end, datetime.fromtimestamp(os.path.getmtime(path))) if end else end
    return (end - start).total_seconds() if start and end else None, frames, answers, ttft


def main():
    print(f"{'method':18s} {'task':9s} {'GPU-min':>8s} {'s/question':>10s} {'frames enc.':>11s} "
          f"{'frames/q':>8s} {'TTFT ms':>8s}  (vs fps0.2 kv6000)")
    base = {}
    for name, tag in TAGS:
        for task in ("sember_mcq", "sember_grounding"):
            logs = sorted(glob.glob(f"{B}/{task}/{tag}/inference-*.log"))
            if not logs:
                continue
            parts = [log_cost(p) for p in logs]
            if any(p[0] is None for p in parts):
                continue
            secs = sum(p[0] for p in parts)
            frames = sum(p[1] for p in parts)
            answers = sum(p[2] for p in parts)
            ttft = [x for p in parts for x in p[3]]
            if name == "fps0.2 kv6000":
                base[task] = secs
            rel = f"x{secs / base[task]:.2f}" if task in base else ""
            print(f"{name:18s} {task[6:]:9s} {secs / 60:8.1f} {secs / max(answers, 1):10.1f} {frames:11d} "
                  f"{frames / max(answers, 1):8.1f} {1000 * st.mean(ttft) if ttft else float('nan'):8.0f}  {rel}")


if __name__ == "__main__":
    main()
