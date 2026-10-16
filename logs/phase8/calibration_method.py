"""Interval-length correction as a grounding method (no GPU).

Every method predicts intervals ~3-4x shorter than the annotated evidence. Three correction rules are
fitted on half of the videos and scored on the other half (2-fold by video, then swapped), so no rule
ever sees its test videos:

- scale+shift: stretch the interval around its centre by alpha and move the centre by beta seconds
- scale:       stretch by alpha only (one parameter)
- min-length:  widen intervals shorter than L seconds to L around their centre; longer ones unchanged

Predictions are clipped to [0, question_time]. Unparseable answers score 0 under every rule.
Run from the repo root: python3 logs/phase8/calibration_method.py
"""
import glob
import hashlib
import json
import os

B = "results/qwen3_vl_8b/sember_grounding"
SUFFIX = "-attention_weighted-native-time-keepsurv"
TAGS = [
    ("uniform 32", "uniform-n32-time-count-location-v300-native-time"),
    ("uniform 64", "uniform-n64-time-count-location-v300-native-time"),
    ("fps0.2 kv6000 (base)", f"time-count-location-fps0.2-kv6000-k0{SUFFIX}-v300"),
    ("fps0.2 kv10700", f"time-count-location-fps0.2-kv10700-k0{SUFFIX}-v300"),
    ("grid16x16 kv6000", f"time-count-location-grid16x16-kv6000-k0{SUFFIX}-v300"),
    ("grid32x8 kv6000", f"time-count-location-grid32x8-kv6000-k0{SUFFIX}-v300"),
    ("grid32x8 kv10700", f"time-count-location-grid32x8-kv10700-k0{SUFFIX}-v300"),
    ("s1x160 kv10700", f"time-count-location-s1x160-kv10700-k0{SUFFIX}-v300"),
    ("dup0.2 sc0.7", f"time-count-location-dup0.2-kv6000-k0{SUFFIX}-scale0.7-v300"),
    ("dups1x60 sc0.7", f"time-count-location-dups1x60-kv6000-k0{SUFFIX}-scale0.7-v300"),
]
# Round 8 runs are picked up automatically once scored.
for d in sorted(glob.glob(f"{B}/time-count-location-grid*d-kv*-scale0.7-v300")) + \
        sorted(glob.glob(f"{B}/time-count-location-dups1x60-kv10700-*-scale0.7-v300")):
    TAGS.append((os.path.basename(d).replace("time-count-location-", "").replace(SUFFIX, ""), os.path.basename(d)))

RULES = {
    "scale+shift": [(a / 4, b) for a in range(2, 21) for b in range(-30, 31, 2)],   # alpha 0.5..5, beta +-30 s
    "scale": [(a / 4, 0) for a in range(2, 21)],
    "min-length": [(L, None) for L in range(0, 121, 5)],                            # L 0..120 s
}


def iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def load(tag):
    rows = []
    for line in open(os.path.join(B, tag, "sember_grounding_scored.jsonl")):
        r = json.loads(line)
        ok = r.get("pred_start_time") not in (None, "") and r.get("pred_end_time") not in (None, "")
        rows.append({"fold": int(hashlib.md5(r["video_id"].encode()).hexdigest(), 16) % 2, "ok": ok,
                     "ps": float(r["pred_start_time"]) if ok else 0.0, "pe": float(r["pred_end_time"]) if ok else 0.0,
                     "gs": float(r["answer_start_time"]), "ge": float(r["answer_end_time"]),
                     "qt": float(r["question_time"])})
    return rows


def apply(r, rule, p):
    c, length = (r["ps"] + r["pe"]) / 2, r["pe"] - r["ps"]
    if rule == "min-length":
        half = max(length, p[0]) / 2
    else:
        c += p[1]
        half = max(length, 1.0) * p[0] / 2
    return max(0.0, c - half), min(r["qt"], c + half)


def score(rows, rule=None, p=None):
    ious = []
    for r in rows:
        if not r["ok"]:
            ious.append(0.0)
            continue
        s, e = (r["ps"], r["pe"]) if rule is None else apply(r, rule, p)
        ious.append(iou(s, e, r["gs"], r["ge"]) if e > s else 0.0)
    return 100 * sum(ious) / len(ious), 100 * sum(x >= 0.5 for x in ious) / len(ious)


def cross_validated(rows, rule):
    total, fits, n = [0.0, 0.0], [], 0
    for fold in (0, 1):
        train = [r for r in rows if r["fold"] != fold]
        test = [r for r in rows if r["fold"] == fold]
        p = max(RULES[rule], key=lambda q: score(train, rule, q)[0])
        fits.append(p[0] if rule != "scale+shift" else p)
        m = score(test, rule, p)
        total = [t + x * len(test) for t, x in zip(total, m)]
        n += len(test)
    return [t / n for t in total], fits


def main():
    print(f"{'method':32s} {'raw':>13s}   " + "   ".join(f"{r:>13s}" for r in RULES) + "   fitted per fold")
    print(f"{'':32s} {'mIoU / R@.5':>13s}   " + "   ".join(f"{'mIoU / R@.5':>13s}" for _ in RULES))
    for name, tag in TAGS:
        if not os.path.exists(os.path.join(B, tag, "sember_grounding_scored.jsonl")):
            continue
        rows = load(tag)
        raw = score(rows)
        cells, fits = [], []
        for rule in RULES:
            m, f = cross_validated(rows, rule)
            cells.append(f"{m[0]:5.1f} / {m[1]:5.1f}")
            fits.append(f"{rule}={f}")
        print(f"{name:32s} {raw[0]:5.1f} / {raw[1]:5.1f}   " + "   ".join(f"{c:>13s}" for c in cells) + "   " + "; ".join(fits))


if __name__ == "__main__":
    main()
