"""Phase 17: coverage-then-zoom vs baselines (Qwen3-VL-8B, S-EMBER grounding, timeline prompt, 475 questions).

For each budget: metrics of the coarse pass (random pruning), zoom with self / gold / whole-video windows,
single-pass random pruning at about twice the budget (matches the two-pass token cost), unpruned 64 frames and the
evidence-window oracle; then paired bootstrap differences of zoom-self against each, and per-task numbers.
Run from the repo root: python3 logs/phase17/compare.py
"""
import sys

sys.path.insert(0, "logs/phase15")
from diagnose_gpu import metrics, paired, rows_of  # noqa: E402

TL = "-timeline"
BUDGETS = {
    "10%": {"coarse (random 10%)": "offline-random-keep0.1" + TL,
            "zoom: self window": "zoom-self-keep0.1-k32-a0.6" + TL,
            "zoom: whole video (control)": "zoom-full-keep0.1-k32-a0.6" + TL,
            "zoom: gold window (oracle)": "zoom-gold-keep0.1-k32-a0.6" + TL,
            "random 25% (2.5x tokens, one pass)": "offline-random-keep0.25" + TL},
    "5%": {"coarse (random 5%)": "offline-random-keep0.05" + TL,
           "zoom: self window": "zoom-self-keep0.05-k32-a0.6" + TL,
           "zoom: gold window (oracle)": "zoom-gold-keep0.05-k32-a0.6" + TL,
           "random 10% (2x tokens, one pass)": "offline-random-keep0.1" + TL},
}
REF = {"unpruned 64 frames": "timeline", "evidence-window oracle": "oracle-window" + TL}
CATS = (("time_duration", "dur"), ("counting_objects_events", "cnt"), ("location_trace", "loc"))


def show(name, rows):
    m = metrics(rows)
    cats = []
    for c, short in CATS:
        sub = {q: r for q, r in rows.items() if r["question_category"] == c}
        cm = metrics(sub)
        cats.append(f"{short} {cm['acc']}/{cm['miou']}/{cm['gq05']}" if cm["acc"] is not None else f"{short} --")
    print(f"  {name:36s} Acc {m['acc']} mIoU {m['miou']} R@.5 {m['r05']} GQ@.5 {m['gq05']} | " + " | ".join(cats))


def main():
    ref = {k: rows_of(v) for k, v in REF.items()}
    for budget, runs in BUDGETS.items():
        print(f"\n=== budget {budget} (per task: acc/mIoU/GQ@.5)")
        loaded = {k: rows_of(v) for k, v in runs.items()}
        for k, rows in list(loaded.items()) + list(ref.items()):
            if rows and all(r["_ok"] is not None for r in rows.values()):
                show(k, rows)
            else:
                print(f"  {k:36s} (not finished or not judged)")
        z = loaded.get("zoom: self window")
        if not z or any(r["_ok"] is None for r in z.values()):
            continue
        for k, rows in list(loaded.items()) + list(ref.items()):
            if k == "zoom: self window" or not rows or any(r["_ok"] is None for r in rows.values()):
                continue
            print(f"  zoom-self minus {k:30s} " + "  ".join(f"{m} {paired(z, rows, m)}" for m in ("acc", "miou", "gq05")))


if __name__ == "__main__":
    main()
