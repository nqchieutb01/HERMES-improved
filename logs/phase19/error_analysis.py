"""Phase 19: error analysis of prior-contrastive decoding (PCD) vs its baseline (random 10%, timeline prompt).

Per question: answer correctness (judge) and IoU before / after PCD. Reports transitions, per-category effects,
whether PCD changed the answer text, the time line or both, and the remaining failure modes after PCD.
Run from the repo root: python3 logs/phase19/error_analysis.py
"""
import collections
import re
import statistics as st
import sys

sys.path.insert(0, "logs/phase15")
sys.path.insert(0, "logs/phase18")
sys.path.insert(0, "logs/phase11")
from report import load  # noqa: E402
import rule_answer as ra  # noqa: E402

SEEN = re.compile(r"^\s*Seen:\s*([\d.:]+)", re.M)
B, M = "offline-random-keep0.1-timeline", "random10-cd-blind-a0.5-timeline"


def answer_text(r):
    m = re.search(r"Answer:\s*(.+?)(?:\n|$)", r["pred_raw"] or "")
    return (m.group(1) if m else "").strip().lower()


def interval(r):
    if not r["interval_parseable"]:
        return None
    return tuple(sorted((float(r["pred_start_time"]), float(r["pred_end_time"]))))


def main():
    b, m = load(B), load(M)
    qs = sorted(set(b) & set(m))
    print(f"questions {len(qs)}")
    tr = collections.Counter((b[q]["_ok"], m[q]["_ok"]) for q in qs)
    print("answer transitions (base, PCD):", dict(tr))
    gq = lambda r: bool(r["_ok"]) and r["_iou"] >= 0.5
    trg = collections.Counter((gq(b[q]), gq(m[q])) for q in qs)
    print("GQ@0.5 transitions:", dict(trg))
    # What changed?
    same_ans = [answer_text(b[q]) == answer_text(m[q]) for q in qs]
    same_int = [interval(b[q]) == interval(m[q]) for q in qs]
    print(f"answer text unchanged {100*st.mean(same_ans):.1f}% | interval unchanged {100*st.mean(same_int):.1f}% | "
          f"both unchanged {100*st.mean(a and i for a, i in zip(same_ans, same_int)):.1f}%")
    for lab, sel in (("answer text changed", [not x for x in same_ans]), ("answer text unchanged", same_ans)):
        sub = [q for q, s in zip(qs, sel) if s]
        print(f"  {lab:22s} n={len(sub):3d} acc {100*st.mean(b[q]['_ok'] for q in sub):5.1f} -> {100*st.mean(m[q]['_ok'] for q in sub):5.1f} | "
              f"mIoU {100*st.mean(b[q]['_iou'] for q in sub):5.1f} -> {100*st.mean(m[q]['_iou'] for q in sub):5.1f}")
    print("\nper category (acc / mIoU / GQ@0.5, base -> PCD):")
    for c in ("time_duration", "counting_objects_events", "location_trace"):
        sub = [q for q in qs if b[q]["question_category"] == c]
        f = lambda R, k: 100 * st.mean((R[q]["_ok"] if k == "acc" else R[q]["_iou"] if k == "miou" else gq(R[q])) for q in sub)
        print(f"  {c:25s} n={len(sub):3d} acc {f(b,'acc'):5.1f}->{f(m,'acc'):5.1f} | mIoU {f(b,'miou'):5.1f}->{f(m,'miou'):5.1f} | GQ {f(b,'gq'):4.1f}->{f(m,'gq'):4.1f}")
    # Duration answers: does the stated duration follow the interval?
    print("\nduration questions: stated duration vs gold and vs own interval length")
    for name, R in (("base", b), ("PCD", m)):
        rows = [R[q] for q in qs if R[q]["question_category"] == "time_duration"]
        within, consistent, n = 0, 0, 0
        for r in rows:
            d = ra.parse_duration(ra.model_answer(r))
            golds = [g for g in (ra.parse_duration(a) for a in ra.annotator_answers(r)) if g]
            iv = interval(r)
            if d is None or not golds or iv is None:
                continue
            n += 1
            within += abs(d - st.median(golds)) <= 0.25 * st.median(golds)
            consistent += abs(d - (iv[1] - iv[0])) <= 0.25 * max(iv[1] - iv[0], 1)
        print(f"  {name:5s} stated within 25% of gold {100*within/n:.1f}% | stated matches own interval {100*consistent/n:.1f}% (n={n})")
    # Counting: count vs gold
    print("\ncounting: under / exact / over (base -> PCD)")
    for name, R in (("base", b), ("PCD", m)):
        c = collections.Counter()
        for q in qs:
            r = R[q]
            if r["question_category"] != "counting_objects_events":
                continue
            k = ra.parse_count(ra.model_answer(r))
            golds = [g for g in (ra.parse_count(a) for a in ra.annotator_answers(r)) if g is not None]
            if k is None or not golds:
                continue
            g = st.median(golds)
            c["under" if k < g else "exact" if k == g else "over"] += 1
        tot = sum(c.values())
        print(f"  {name:5s} " + " ".join(f"{k} {100*v/tot:.1f}%" for k, v in sorted(c.items())))
    # Failure modes after PCD
    print("\nfailure modes after PCD (share of all questions):")
    fm = collections.Counter()
    for q in qs:
        r = m[q]
        raw = r["pred_raw"] or ""
        iv = interval(r)
        if "Answer:" not in raw:
            fm["no answer (list runs to token limit)"] += 1
        elif iv is None:
            fm["interval unparsed (m:ss or missing)"] += 1
        elif r["_iou"] >= 0.5:
            fm["IoU>=0.5"] += 1
        else:
            gs, ge = r["_gs"], r["_ge"]
            s, e = iv
            if e < gs:
                fm["disjoint, before gold"] += 1
            elif s > ge:
                fm["disjoint, after gold"] += 1
            elif s >= gs and e <= ge:
                fm["inside gold, too short"] += 1
            elif s <= gs and e >= ge:
                fm["covers gold, too long"] += 1
            else:
                fm["partial overlap"] += 1
    for k, v in fm.most_common():
        print(f"  {k:38s} {100*v/len(qs):5.1f}%")
    seen_n = [len(SEEN.findall(m[q]["pred_raw"] or "")) for q in qs]
    seen_b = [len(SEEN.findall(b[q]["pred_raw"] or "")) for q in qs]
    print(f"\nmedian # Seen lines base {st.median(seen_b)} -> PCD {st.median(seen_n)}; >8 lines: base {sum(x > 8 for x in seen_b)} PCD {sum(x > 8 for x in seen_n)}")


if __name__ == "__main__":
    main()
