"""Post-hoc interval calibration for S-EMBER grounding (no GPU).

Every method predicts intervals that are too short (median 12-18 s vs 43 s gold). Fit a length scale
alpha and a centre shift beta (seconds) on half the videos, apply to the other half (2-fold by video),
and report calibrated mIoU / R@0.5. Predictions are clipped to [0, question_time]. Shows how much of
the grounding gap is interval sizing rather than memory, and whether method rankings change.
Run from the repo root: python3 logs/phase7/calibrate.py
"""
import hashlib
import json
import os

B = "results/qwen3_vl_8b/sember_grounding"
TAGS = [
    ("uniform 32", "uniform-n32-time-count-location-v300-native-time"),
    ("uniform 64", "uniform-n64-time-count-location-v300-native-time"),
    ("fps0.2 kv6000", "time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("fps0.2 kv10700", "time-count-location-fps0.2-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
    ("s1x60 kv6000", "time-count-location-s1x60-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("s1x160 kv10700", "time-count-location-s1x160-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid16x16 kv6000", "time-count-location-grid16x16-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid32x8 kv6000", "time-count-location-grid32x8-kv6000-k0-attention_weighted-native-time-keepsurv-v300"),
    ("grid32x8 kv10700", "time-count-location-grid32x8-kv10700-k0-attention_weighted-native-time-keepsurv-v300"),
]
ALPHAS = [0.5 * i for i in range(1, 25)]           # 0.5 .. 12.0
BETAS = [-30 + 2 * i for i in range(31)]          # -30 .. +30 s


def iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def load(tag):
    rows = []
    for line in open(os.path.join(B, tag, "sember_grounding_scored.jsonl")):
        r = json.loads(line)
        ok = r.get("pred_start_time") not in (None, "") and r.get("pred_end_time") not in (None, "")
        rows.append({
            "fold": int(hashlib.md5(r["video_id"].encode()).hexdigest(), 16) % 2,
            "ok": ok,
            "ps": float(r["pred_start_time"]) if ok else 0.0,
            "pe": float(r["pred_end_time"]) if ok else 0.0,
            "gs": float(r["answer_start_time"]),
            "ge": float(r["answer_end_time"]),
            "qt": float(r["question_time"]),
        })
    return rows


def score(rows, alpha, beta):
    ious = []
    for r in rows:
        if not r["ok"]:
            ious.append(0.0)
            continue
        c = (r["ps"] + r["pe"]) / 2 + beta
        half = max(r["pe"] - r["ps"], 1.0) * alpha / 2
        s, e = max(0.0, c - half), min(r["qt"], c + half)
        ious.append(iou(s, e, r["gs"], r["ge"]) if e > s else 0.0)
    return 100 * sum(ious) / len(ious), 100 * sum(x >= 0.5 for x in ious) / len(ious)


def main():
    print(f"{'method':18s} {'raw mIoU':>8s} {'raw R@.5':>8s}   {'cal mIoU':>8s} {'cal R@.5':>8s}   fitted (alpha, beta) per fold")
    for name, tag in TAGS:
        if not os.path.exists(os.path.join(B, tag, "sember_grounding_scored.jsonl")):
            continue
        rows = load(tag)
        raw = score(rows, 1.0, 0.0)
        held, fits = [], []
        for fold in (0, 1):
            train = [r for r in rows if r["fold"] != fold]
            test = [r for r in rows if r["fold"] == fold]
            a, b = max(((a, b) for a in ALPHAS for b in BETAS), key=lambda ab: score(train, *ab)[0])
            fits.append((a, b))
            held.append((score(test, a, b), len(test)))
        n = sum(k for _, k in held)
        cal = [sum(s[i] * k for s, k in held) / n for i in (0, 1)]
        print(f"{name:18s} {raw[0]:8.1f} {raw[1]:8.1f}   {cal[0]:8.1f} {cal[1]:8.1f}   {fits}")


if __name__ == "__main__":
    main()
