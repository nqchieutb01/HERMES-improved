"""Free analysis: where does Qwen3 HERMES streaming lose to uniform sampling on S-EMBER grounding?"""
import json, statistics as st
B = "results/qwen3_vl_8b/sember_grounding/"
RUNS = {
    "stream KV6000": "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300",
    "stream KV10700": "time-count-location-fps0.2-kv10700-k0-attention_weighted-native-time-keepsurv-v300",
    "stream KV21400": "time-count-location-fps0.2-kv21400-k0-attention_weighted-native-time-v300",
    "uniform 32": "uniform-n32-time-count-location-v300-native-time",
    "uniform 64": "uniform-n64-time-count-location-v300-native-time",
}
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
data = {k: {r["question_id"]: r for r in map(json.loads, open(B + v + "/sember_grounding_scored.jsonl"))} for k, v in RUNS.items()}
qids = sorted(set.intersection(*(set(d) for d in data.values())))
ref = data["uniform 32"]
def miou(run, qs): return 100 * st.mean(f(data[run][q]["temporal_iou"]) or 0 for q in qs) if qs else float("nan")
def table(title, key, bins):
    print(f"\n== {title}  (mean IoU %)")
    print(f"{'bucket':>14s} {'n':>4s} " + " ".join(f"{k:>15s}" for k in RUNS) + "   stream6000-uni32")
    for lo, hi in bins:
        qs = [q for q in qids if key(ref[q]) is not None and lo <= key(ref[q]) < hi]
        s, u = miou("stream KV6000", qs), miou("uniform 32", qs)
        print(f"{f'{lo:g}-{hi:g}':>14s} {len(qs):4d} " + " ".join(f"{miou(k, qs):15.1f}" for k in RUNS) + f"   {s - u:+6.1f}")
table("by question time (s): how much video has streamed in", lambda r: f(r["question_time"]), [(0, 120), (120, 240), (240, 360), (360, 480), (480, 1e9)])
table("by memory recency (s): how long ago the evidence ended", lambda r: f(r["memory_recency"]), [(0, 30), (30, 90), (90, 180), (180, 300), (300, 1e9)])
table("by evidence length (s)", lambda r: (f(r["answer_end_time"]) or 0) - (f(r["answer_start_time"]) or 0), [(0, 10), (10, 30), (30, 90), (90, 1e9)])
# streaming frames seen at question time vs uniform's 32
table("by frames streamed at question (0.2 fps) vs uniform's 32", lambda r: f(r["question_time"]) * 0.2, [(0, 16), (16, 32), (32, 64), (64, 1e9)])

print("\n== predicted interval vs correct interval (parseable answers)")
print(f"{'run':>15s} {'parse%':>7s} {'pred len med':>12s} {'gold len med':>12s} {'center err med':>14s} {'pred ends after q':>18s} {'no overlap':>11s} {'IoU=0 but <30s off':>19s}")
for k in RUNS:
    rows = [data[k][q] for q in qids]
    ok = [r for r in rows if f(r["pred_start_time"]) is not None and f(r["pred_end_time"]) is not None]
    pl = [f(r["pred_end_time"]) - f(r["pred_start_time"]) for r in ok]
    gl = [f(r["answer_end_time"]) - f(r["answer_start_time"]) for r in ok]
    ce = [abs((f(r["pred_end_time"]) + f(r["pred_start_time"])) / 2 - (f(r["answer_end_time"]) + f(r["answer_start_time"])) / 2) for r in ok]
    after = sum(f(r["pred_end_time"]) > f(r["question_time"]) + 1 for r in ok)
    zero = [r for r in ok if (f(r["temporal_iou"]) or 0) == 0]
    near = sum(min(abs(f(r["pred_start_time"]) - f(r["answer_end_time"])), abs(f(r["answer_start_time"]) - f(r["pred_end_time"]))) < 30 for r in zero)
    print(f"{k:>15s} {100*len(ok)/len(rows):7.1f} {st.median(pl):12.1f} {st.median(gl):12.1f} {st.median(ce):14.1f} {100*after/len(ok):17.1f}% {100*len(zero)/len(ok):10.1f}% {100*near/max(len(zero),1):18.1f}%")

print("\n== same-question wins: stream KV6000 vs uniform 32")
s = {q: f(data["stream KV6000"][q]["temporal_iou"]) or 0 for q in qids}; u = {q: f(ref[q]["temporal_iou"]) or 0 for q in qids}
print(f"stream better {sum(s[q] > u[q] + 0.05 for q in qids)}, uniform better {sum(u[q] > s[q] + 0.05 for q in qids)}, within 0.05: {sum(abs(s[q]-u[q]) <= 0.05 for q in qids)}  (of {len(qids)})")
