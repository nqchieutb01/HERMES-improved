"""Share of kept visual tokens that fall inside the gold evidence interval, per token selector and budget.

S-EMBER grounding, Qwen3-VL-8B, 64 uniform frames unless noted; 475 questions (those with at least one frame in
the gold interval and one outside). For each question: precision = kept tokens on gold frames / all kept
tokens; recall = kept tokens on gold frames / encoded tokens on gold frames; gold frames left with no token.
Averaged over questions.
  - Pruning selectors (HERMES, random, stratified, oracle, hermes_exact): from retention-<chunk>.jsonl logs.
  - Unpruned and spatial pooling: every frame keeps all its (fewer) tokens, so precision = share of the 64
    frames inside the gold interval; recall = the keep ratio for pooling.
  - Uniform frame dropping (N frames): exact from the sampled frame times; each kept frame keeps all tokens.
Run from the repo root: python3 logs/phase16/evidence_share.py
"""
import glob
import json
import statistics as st

G = "results/qwen3_vl_8b/sember_grounding/"
U = "uniform-n64-time-count-location-"
RET = {
    "HERMES 50%": "dx-ret-hermes-keep0.5", "HERMES 25%": "dx-ret-hermes-keep0.25",
    "HERMES 10%": "dx-ret-hermes-keep0.1", "HERMES 5%": "dx-ret-hermes-keep0.05",
    "HERMES-exact 10%": "offline-hermes_exact-keep0.1", "HERMES-exact 5%": "offline-hermes_exact-keep0.05",
    "Random 10%": "dx-ret-random-keep0.1", "Stratified 10%": "dx-ret-stratified-keep0.1",
    "Oracle 10%": "offline-oracle-keep0.1", "Oracle 5%": "offline-oracle-keep0.05",
}


def gold_of(tag):
    rows = [json.loads(l) for l in open(G + U + tag + "-v300-native-time/sember_grounding_scored.jsonl")] \
        if tag else [json.loads(l) for l in open(G + U + "v300-native-time/sember_grounding_scored.jsonl")]
    return {r["question_id"]: r for r in rows}


def from_retention(tag):
    try:
        rows = gold_of(tag)
    except FileNotFoundError:
        return None
    prec, rec, dead = [], [], []
    for f in glob.glob(G + U + tag + "-v300-native-time/retention-*.jsonl"):
        for l in open(f):
            s = json.loads(l)
            r = rows.get(s["question_id"])
            if not r:
                continue
            gs, ge = float(r["answer_start_time"]), float(r["answer_end_time"])
            gold = {i for i, t in enumerate(s["frame_times"]) if gs <= t <= ge}
            if not gold or len(gold) == len(s["frame_times"]):
                continue
            kept = {int(k): v for k, v in s["kept"].items()}
            enc = {int(k): v for k, v in s["encoded"].items()}
            tot = sum(kept.values())
            prec.append(sum(kept.get(i, 0) for i in gold) / tot)
            rec.append(sum(kept.get(i, 0) for i in gold) / sum(enc.get(i, 0) for i in gold))
            dead.append(sum(kept.get(i, 0) < 1 for i in gold) / len(gold))
    if not prec:
        return None
    return len(prec), st.mean(prec), st.mean(rec), st.mean(dead)


def frame_times(r, n):
    """Times of n uniform frames over [0, question time] (same integer rule as uniform_frame_indices)."""
    idx = json.loads(r["source_frame_indices_json"])
    fps = idx[-1] / max(float(r["question_time"]), 1e-6) if idx[-1] else 1.0
    end = idx[-1] + 1
    count = min(n, end)
    return [((k * (end - 1)) // max(count - 1, 1)) / fps for k in range(count)]


def uniform_frames(n, keep_ratio_of_64):
    rows = gold_of("")
    prec, rec, dead = [], [], []
    for r in rows.values():
        gs, ge = float(r["answer_start_time"]), float(r["answer_end_time"])
        t64 = frame_times(r, 64)
        g64 = [gs <= t <= ge for t in t64]
        if not any(g64) or all(g64):
            continue
        t = frame_times(r, n)
        g = [gs <= x <= ge for x in t]
        prec.append(sum(g) / len(t))
        # Recall relative to the 64-frame encoding: kept gold frames / gold frames among the 64.
        rec.append(min(1.0, sum(g) / sum(g64)))
        dead.append(1.0 if not any(g) else 0.0)
    return len(prec), st.mean(prec), st.mean(rec), st.mean(dead)


def main():
    print(f"{'selector':22s} {'n':>4s} {'% kept tokens in gold':>22s} {'% gold tokens kept':>19s} {'no gold frame kept*':>20s}")
    base = uniform_frames(64, 1.0)
    print(f"{'Unpruned (64 frames)':22s} {base[0]:4d} {100 * base[1]:22.1f} {100.0:19.1f} {'0.0':>20s}")
    for s, k in ((0.7, 0.49), (0.5, 0.25), (0.35, 0.12)):
        print(f"{f'Pooling x{s} (~{int(100 * k)}%)':22s} {base[0]:4d} {100 * base[1]:22.1f} {100 * k:19.1f} {'0.0':>20s}")
    for n, k in ((32, 50), (16, 25), (12, 19), (6, 9), (4, 6)):
        v = uniform_frames(n, k / 100)
        print(f"{f'Uniform {n} frames (~{k}%)':22s} {v[0]:4d} {100 * v[1]:22.1f} {100 * v[2]:19.1f} {100 * v[3]:19.1f}*")
    for name, tag in RET.items():
        v = from_retention(tag)
        if v is None:
            print(f"{name:22s}  (retention log not available yet)")
            continue
        print(f"{name:22s} {v[0]:4d} {100 * v[1]:22.1f} {100 * v[2]:19.1f} {100 * v[3]:20.1f}")
    print("* uniform frame dropping: % of questions with no sampled frame inside the gold interval; "
          "other rows: % of gold frames left with no token")


if __name__ == "__main__":
    main()
