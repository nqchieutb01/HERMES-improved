"""Free analysis (no GPU): where does Qwen3 streaming lose to uniform sampling on S-EMBER grounding (v300)?

Splits mIoU / R@0.5 by question time, video length and whether the KV cache had already been
compressed before the question, and measures how predicted intervals are biased (length, shift).
Compression state is recovered from the streaming runs' inference logs: HERMES logs
"Applying KV-Cache compression" whenever the cache exceeds the budget, and "Pred Answer:" once per
question, in the same order as the rows of <chunks>_<idx>.csv.

Run from the repo root: python logs/grounding_breakdown.py
"""
import csv
import json
import os
import re
import statistics as st

B = "results/qwen3_vl_8b/sember_grounding/"
RUNS = {
    "stream KV6000": "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300",
    "stream KV21400": "time-count-location-fps0.2-kv21400-k0-attention_weighted-native-time-keepsurv-v300",
    "uniform 32": "uniform-n32-time-count-location-v300-native-time",
    "uniform 64": "uniform-n64-time-count-location-v300-native-time",
}
STREAM, UNI = "stream KV6000", "uniform 32"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compression_flags_by_video(run_dir):
    """question_id -> True if the cache was compressed at least once for this video before answering."""
    flags = {}
    for i in range(4):
        rows = list(csv.DictReader(open(os.path.join(run_dir, f"4_{i}.csv"))))
        states, compressed, answered_since_reset = [], False, False
        prev_vid, row_idx = None, 0
        for line in open(os.path.join(run_dir, f"inference-{i}.log"), errors="replace"):
            if line.startswith("Encoding frames 0 to") and row_idx < len(rows):
                vid = rows[row_idx]["video_id"]
                if vid != prev_vid:
                    compressed, prev_vid = False, vid
            if "Applying KV-Cache compression" in line:
                compressed = True
            if line.startswith("Pred Answer:"):
                states.append(compressed)
                row_idx += 1
        if len(states) != len(rows):
            raise RuntimeError(f"{run_dir} chunk {i}: {len(states)} answers in log vs {len(rows)} rows")
        for r, s in zip(rows, states):
            flags[r["question_id"]] = s
    return flags


data = {k: {r["question_id"]: r for r in map(json.loads, open(B + v + "/sember_grounding_scored.jsonl"))}
        for k, v in RUNS.items()}
qids = sorted(set.intersection(*(set(d) for d in data.values())))
ref = data[UNI]
comp = {k: compression_flags_by_video(B + v) for k, v in RUNS.items() if k.startswith("stream")}


def iou(run, q):
    return f(data[run][q]["temporal_iou"]) or 0.0


def r05(run, q):
    return 1.0 if data[run][q]["recall_at_1_iou_0.5"] in (True, "True", "true", 1) else 0.0


def split(title, key, bins):
    print(f"\n== {title}")
    print(f"{'bucket':>16s} {'n':>4s} " + " ".join(f"{k:>15s}" for k in RUNS) + f"   {'stream6k-uni32':>14s}")
    print(f"{'':>16s} {'':>4s} " + " ".join(f"{'mIoU / R@.5':>15s}" for _ in RUNS))
    for label, pred in bins:
        qs = [q for q in qids if key(q) is not None and pred(key(q))]
        if not qs:
            continue
        cells = [f"{100 * st.mean(iou(k, q) for q in qs):6.1f} / {100 * st.mean(r05(k, q) for q in qs):5.1f}"
                 for k in RUNS]
        d = 100 * (st.mean(iou(STREAM, q) for q in qs) - st.mean(iou(UNI, q) for q in qs))
        print(f"{label:>16s} {len(qs):4d} " + " ".join(f"{c:>15s}" for c in cells) + f"   {d:+14.1f}")


def rng(lo, hi, unit=""):
    return (f"{lo:g}-{hi:g}{unit}" if hi < 1e8 else f">={lo:g}{unit}", lambda x: lo <= x < hi)


# 1. Question time (how much video has streamed in); at 0.2 fps, frames = t / 5 vs uniform's fixed 32.
split("by question time (s)  [streamed frames at 0.2 fps = t/5; uniform always 32]",
      lambda q: f(ref[q]["question_time"]),
      [rng(0, 80, "s"), rng(80, 160, "s"), rng(160, 240, "s"), rng(240, 320, "s"), rng(320, 480, "s"), rng(480, 1e9, "s")])

# 3a. Video length.
split("by video length (s)", lambda q: f(ref[q]["duration"]),
      [rng(0, 320, "s"), rng(320, 500, "s"), rng(500, 1e9, "s")])

# 3b. Was the KV6000 cache compressed before this question?
split("by KV6000 cache compressed before the question",
      lambda q: comp[STREAM][q], [("no", lambda x: not x), ("yes", lambda x: x)])
split("by KV6000 compressed x question time", lambda q: (comp[STREAM][q], f(ref[q]["question_time"])),
      [(f"{'comp' if c else 'raw'} {lo}-{hi}s" if hi < 1e8 else f"{'comp' if c else 'raw'} >={lo}s",
        (lambda c, lo, hi: lambda x: x[0] == c and lo <= x[1] < hi)(c, lo, hi))
       for c in (False, True) for lo, hi in ((0, 160), (160, 320), (320, 1e9))])
split("by KV21400 cache compressed before the question",
      lambda q: comp["stream KV21400"][q], [("no", lambda x: not x), ("yes", lambda x: x)])

# 2. Predicted interval length and position, relative to the correct interval.
print("\n== predicted vs correct interval (questions where every run gave a parseable interval)")
ok = [q for q in qids if all(f(data[k][q]["pred_start_time"]) is not None and f(data[k][q]["pred_end_time"]) is not None
                             for k in RUNS)]
print(f"n = {len(ok)} of {len(qids)}")
hdr = ["pred len", "gold len", "len ratio", "start err", "end err", "center err", "|center err|",
       "too long", "too short", "early", "late", "IoU=0"]
print(f"{'run':>15s} " + " ".join(f"{h:>12s}" for h in hdr))
print(f"{'':>15s} " + " ".join(f"{s:>12s}" for s in ["med s", "med s", "med x", "med s", "med s", "med s", "med s",
                                                         "% >2x", "% <0.5x", "% ctr<-15s", "% ctr>+15s", "%"]))
for k in RUNS:
    P = [(f(data[k][q]["pred_start_time"]), f(data[k][q]["pred_end_time"])) for q in ok]
    G = [(f(data[k][q]["answer_start_time"]), f(data[k][q]["answer_end_time"])) for q in ok]
    pl = [e - s for s, e in P]
    gl = [e - s for s, e in G]
    ratio = [max(p, 1) / max(g, 1) for p, g in zip(pl, gl)]
    ds = [p[0] - g[0] for p, g in zip(P, G)]
    de = [p[1] - g[1] for p, g in zip(P, G)]
    dc = [(p[0] + p[1]) / 2 - (g[0] + g[1]) / 2 for p, g in zip(P, G)]
    zero = sum(iou(k, q) == 0 for q in ok)
    vals = [st.median(pl), st.median(gl), st.median(ratio), st.median(ds), st.median(de), st.median(dc),
            st.median(abs(x) for x in dc), 100 * sum(r > 2 for r in ratio) / len(ok),
            100 * sum(r < 0.5 for r in ratio) / len(ok), 100 * sum(x < -15 for x in dc) / len(ok),
            100 * sum(x > 15 for x in dc) / len(ok), 100 * zero / len(ok)]
    print(f"{k:>15s} " + " ".join(f"{v:12.1f}" for v in vals))

# Center error signed, by question time: does streaming drift late/early as more video streams in?
print("\n== median signed center error (pred - gold, s) by question time; + = too late")
print(f"{'bucket':>16s} " + " ".join(f"{k:>15s}" for k in RUNS))
for label, pred in [rng(0, 160, "s"), rng(160, 320, "s"), rng(320, 1e9, "s")]:
    qs = [q for q in ok if pred(f(ref[q]["question_time"]))]
    cells = []
    for k in RUNS:
        dc = [(f(data[k][q]["pred_start_time"]) + f(data[k][q]["pred_end_time"])) / 2
              - (f(data[k][q]["answer_start_time"]) + f(data[k][q]["answer_end_time"])) / 2 for q in qs]
        cells.append(f"{st.median(dc):+.1f} (n={len(qs)})")
    print(f"{label:>16s} " + " ".join(f"{c:>15s}" for c in cells))

print("\n== same-question comparison, stream KV6000 vs uniform 32 (IoU margin 0.05)")
s = {q: iou(STREAM, q) for q in qids}
u = {q: iou(UNI, q) for q in qids}
print(f"stream better {sum(s[q] > u[q] + 0.05 for q in qids)}, uniform better {sum(u[q] > s[q] + 0.05 for q in qids)}, "
      f"tie {sum(abs(s[q] - u[q]) <= 0.05 for q in qids)}  (of {len(qids)})")
