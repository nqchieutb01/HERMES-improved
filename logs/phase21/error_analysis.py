"""Phase 21: error analysis of the final method (adaptive, time-scoped PCD) against the baseline on all 8 memories.

Pools the 8 memories (random 5/10/25%, unpruned, HERMES 10%, stratified 10%, streaming 4k/6k; 475 questions each).
Paired differences use a cluster bootstrap over questions (the 8 copies of a question move together).
Sections: failure modes; question type; evidence position, length and frame coverage; annotator agreement; what the
method fixes and breaks; answer-interval consistency; per-memory failure modes.
Writes logs/phase21/error_analysis.md. Run from the repo root: python3 logs/phase21/error_analysis.py
"""
import collections
import json
import random
import re
import statistics as st
import sys

sys.path.insert(0, "logs/phase15")
sys.path.insert(0, "logs/phase18")
sys.path.insert(0, "logs/phase11")
from report import S, load  # noqa: E402
import rule_answer as ra  # noqa: E402

F = "cd-blind-adapt1.0-scopetime-timeline"
MEMORIES = [
    ("Random 5%", "offline-random-keep0.05-timeline", f"random5-{F}"),
    ("Random 10%", "offline-random-keep0.1-timeline", f"random10-{F}"),
    ("Random 25%", "offline-random-keep0.25-timeline", f"random25-{F}"),
    ("Unpruned", "timeline", f"unpruned-{F}"),
    ("HERMES 10%", "offline-hermes-keep0.1-timeline", f"hermes10-{F}"),
    ("Stratified 10%", "offline-stratified-keep0.1-timeline", f"stratified10-{F}"),
    ("Streaming 4k", "@" + S.format(4000, "timeline-"), "@" + S.format(4000, F + "-")),
    ("Streaming 6k", "@" + S.format(6000, "timeline-"), "@" + S.format(6000, F + "-")),
]
SEEN = re.compile(r"^\s*Seen:\s*([\d.:]+)", re.M)
CATS = {"time_duration": "duration", "counting_objects_events": "counting", "location_trace": "location"}


def interval(r):
    if not r["interval_parseable"]:
        return None
    return tuple(sorted((float(r["pred_start_time"]), float(r["pred_end_time"]))))


def gq(r):
    return bool(r["_ok"]) and r["_iou"] >= 0.5


def failure_mode(r):
    raw = r["pred_raw"] or ""
    iv = interval(r)
    if "Answer:" not in raw:
        return "no answer line (list runs on)"
    if iv is None:
        return "interval missing or unparsed"
    if r["_iou"] >= 0.5:
        return "IoU>=0.5, answer correct" if r["_ok"] else "IoU>=0.5, answer wrong"
    s, e = iv
    gs, ge = r["_gs"], r["_ge"]
    if e < gs:
        return "disjoint, before gold"
    if s > ge:
        return "disjoint, after gold"
    if s >= gs and e <= ge:
        return "inside gold, too short"
    if s <= gs and e >= ge:
        return "covers gold, too long"
    return "partial overlap"


def annotator_iou(r):
    """Mean pairwise IoU of the annotators' intervals (label agreement); None with fewer than two intervals."""
    ivs = [(float(a["start_ts"]), float(a["end_ts"])) for a in json.loads(r.get("answers_json") or "[]")
           if a.get("start_ts") is not None and a.get("end_ts") is not None]
    if len(ivs) < 2:
        return None
    vals = []
    for i in range(len(ivs)):
        for j in range(i + 1, len(ivs)):
            (a, b), (c, d) = ivs[i], ivs[j]
            inter = max(0.0, min(b, d) - max(a, c))
            union = max(b, d) - min(a, c)
            vals.append(inter / union if union > 0 else 1.0)
    return st.mean(vals)


def cluster_delta(pairs, f, n=2000, seed=0):
    """Mean of f(method) - f(base) over pairs, with a 95% cluster bootstrap over question ids."""
    by_q = collections.defaultdict(list)
    for q, b, m in pairs:
        by_q[q].append(f(m) - f(b))
    qs = list(by_q)
    tot = sum(len(v) for v in by_q.values())
    mean = sum(sum(v) for v in by_q.values()) / tot
    rng, boots = random.Random(seed), []
    for _ in range(n):
        sample = [by_q[rng.choice(qs)] for _ in qs]
        boots.append(sum(sum(v) for v in sample) / sum(len(v) for v in sample))
    boots.sort()
    lo, hi = boots[int(0.025 * n)], boots[int(0.975 * n)]
    return f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else "")


def pct(xs):
    return 100.0 * sum(xs) / max(len(xs), 1)


def group_table(pairs, key, order, title, out):
    out += [f"## {title}", "", "| Group | n | Acc. base → final | mIoU base → final | GQ@0.5 base → final | ΔmIoU | ΔGQ@0.5 |",
            "|---|---|---|---|---|---|---|"]
    groups = collections.defaultdict(list)
    for p in pairs:
        k = key(p[1])
        if k is not None:
            groups[k].append(p)
    for g in order:
        ps = groups.get(g, [])
        if not ps:
            continue
        b, m = [p[1] for p in ps], [p[2] for p in ps]
        out.append(f"| {g} | {len(ps) // len(MEMORIES)} q × 8 | {pct([bool(r['_ok']) for r in b]):.1f} → "
                   f"{pct([bool(r['_ok']) for r in m]):.1f} | {100 * st.mean(r['_iou'] for r in b):.1f} → "
                   f"{100 * st.mean(r['_iou'] for r in m):.1f} | {pct([gq(r) for r in b]):.1f} → {pct([gq(r) for r in m]):.1f} | "
                   f"{cluster_delta(ps, lambda r: 100 * r['_iou'])} | {cluster_delta(ps, lambda r: 100.0 * gq(r))} |")
    out.append("")


def main():
    pairs, per_memory = [], {}
    for name, btag, ftag in MEMORIES:
        b, m = load(btag), load(ftag)
        assert b is not None and m is not None, name
        qs = sorted(set(b) & set(m))
        per_memory[name] = [(q, b[q], m[q]) for q in qs]
        pairs += per_memory[name]
    out = ["# Error analysis of the final method (adaptive, time-scoped PCD) vs the baseline", "",
           "Generated by `logs/phase21/error_analysis.py`. All 8 memories pooled (3,800 question instances = 475 "
           "questions × 8). Δ = final − baseline; 95% cluster bootstrap over questions; * = excludes 0.", ""]

    # 1. Failure modes
    modes = ["IoU>=0.5, answer correct", "IoU>=0.5, answer wrong", "partial overlap", "inside gold, too short",
             "covers gold, too long", "disjoint, before gold", "disjoint, after gold", "interval missing or unparsed",
             "no answer line (list runs on)"]
    out += ["## 1. Failure modes (share of all question instances)", "",
            "| Outcome | Baseline | Final | Δ |", "|---|---|---|---|"]
    for mode in modes:
        out.append(f"| {mode} | {pct([failure_mode(b) == mode for _, b, _ in pairs]):.1f}% | "
                   f"{pct([failure_mode(m) == mode for _, _, m in pairs]):.1f}% | "
                   f"{cluster_delta(pairs, lambda r, mode=mode: 100.0 * (failure_mode(r) == mode))} |")
    out.append("")
    # Size of the errors that remain.
    short = [(m, interval(m)) for _, _, m in pairs if failure_mode(m) == "inside gold, too short"]
    ratio = [(iv[1] - iv[0]) / max(m["_ge"] - m["_gs"], 1e-6) for m, iv in short]
    single = [iv[1] - iv[0] < 1.0 for _, iv in short]
    before = [(m, interval(m)) for _, _, m in pairs if failure_mode(m) == "disjoint, before gold"]
    gap = [m["_gs"] - iv[1] for m, iv in before]
    late = [m["_gs"] >= 0.5 * m["_qt"] for m, _ in before]
    after = [(m, interval(m)) for _, _, m in pairs if failure_mode(m) == "disjoint, after gold"]
    out += [f"- Too short: predicted length is a median {100 * st.median(ratio):.0f}% of the gold length; "
            f"{pct(single):.0f}% are a single moment (< 1 s).",
            f"- Disjoint before gold: median gap {st.median(gap):.0f} s; {pct(late):.0f}% have their evidence in the "
            f"second half of the window (an earlier, similar event is chosen).",
            f"- Disjoint after gold: {len(after)} instances; median gap "
            f"{st.median([iv[0] - m['_ge'] for m, iv in after]):.0f} s.", ""]

    # 2. Question type
    group_table(pairs, lambda r: CATS.get(r["question_category"]), ["duration", "counting", "location"],
                "2. By question type", out)

    # 3. Evidence position, length and coverage
    def position(r):
        x = r["_gs"] / max(r["_qt"], 1e-6)
        return "start < 5%" if x < 0.05 else "5–33%" if x < 1 / 3 else "33–66%" if x < 2 / 3 else "66–100%"

    group_table(pairs, position, ["start < 5%", "5–33%", "33–66%", "66–100%"],
                "3a. By where the evidence starts (share of the question window)", out)

    def length(r):
        x = (r["_ge"] - r["_gs"]) / max(r["_qt"], 1e-6)
        return "< 2% of window" if x < 0.02 else "2–10%" if x < 0.10 else "10–30%" if x < 0.30 else "> 30%"

    group_table(pairs, length, ["< 2% of window", "2–10%", "10–30%", "> 30%"],
                "3b. By evidence length (share of the question window)", out)

    def coverage(r):
        # Uniform frames (of 64) that fall inside the gold interval: how much of the evidence the model could see.
        k = 64 * (r["_ge"] - r["_gs"]) / max(r["_qt"], 1e-6)
        return "< 1 frame" if k < 1 else "1–3 frames" if k < 4 else "4–15 frames" if k < 16 else "≥ 16 frames"

    group_table(pairs, coverage, ["< 1 frame", "1–3 frames", "4–15 frames", "≥ 16 frames"],
                "3c. By uniform frames inside the evidence (of 64 sampled; offline memories, before pruning)", out)

    # 4. Annotator agreement
    def agreement(r):
        a = annotator_iou(r)
        return None if a is None else "annotators IoU < 0.3" if a < 0.3 else "0.3–0.6" if a < 0.6 else "≥ 0.6"

    group_table(pairs, agreement, ["annotators IoU < 0.3", "0.3–0.6", "≥ 0.6"],
                "4. By annotator agreement on the interval (mean pairwise IoU of the human intervals)", out)

    # 5. What the method fixes and what it breaks
    tr = collections.Counter((failure_mode(b), failure_mode(m)) for _, b, m in pairs)
    fixed = sum(v for (a, c), v in tr.items() if not a.startswith("IoU>=0.5") and c.startswith("IoU>=0.5"))
    broken = sum(v for (a, c), v in tr.items() if a.startswith("IoU>=0.5") and not c.startswith("IoU>=0.5"))
    out += ["## 5. Transitions: what the method fixes and breaks (IoU ≥ 0.5)", "",
            f"- Fixed (baseline IoU < 0.5 → final ≥ 0.5): {fixed} instances ({100 * fixed / len(pairs):.1f}%); "
            f"broken (≥ 0.5 → < 0.5): {broken} ({100 * broken / len(pairs):.1f}%).", "",
            "| Baseline outcome → final outcome | instances |", "|---|---|"]
    for (a, c), v in tr.most_common(14):
        if a != c:
            out.append(f"| {a} → {c} | {v} |")
    out.append("")
    early_b = pct([(interval(b) or (1e9, 0))[0] < 0.05 * b["_qt"] for _, b, _ in pairs if b["_gs"] >= 0.05 * b["_qt"]])
    early_m = pct([(interval(m) or (1e9, 0))[0] < 0.05 * m["_qt"] for _, _, m in pairs if m["_gs"] >= 0.05 * m["_qt"]])
    out += [f"- Prior leak (interval starts in the first 5% although the evidence starts later): {early_b:.1f}% → "
            f"{early_m:.1f}% of those instances.", ""]

    # 6. Answer-interval consistency and answer errors
    out += ["## 6. Answers: duration and counting", ""]
    for label, idx in (("baseline", 1), ("final", 2)):
        dur = [p[idx] for p in pairs if p[idx]["question_category"] == "time_duration"]
        within = consistent = n = 0
        for r in dur:
            d = ra.parse_duration(ra.model_answer(r))
            golds = [g for g in (ra.parse_duration(a) for a in ra.annotator_answers(r)) if g]
            iv = interval(r)
            if d is None or not golds or iv is None:
                continue
            n += 1
            within += abs(d - st.median(golds)) <= 0.25 * st.median(golds)
            consistent += abs(d - (iv[1] - iv[0])) <= 0.25 * max(iv[1] - iv[0], 1)
        cnt = collections.Counter()
        for p in pairs:
            r = p[idx]
            if r["question_category"] != "counting_objects_events":
                continue
            k = ra.parse_count(ra.model_answer(r))
            golds = [g for g in (ra.parse_count(a) for a in ra.annotator_answers(r)) if g is not None]
            if k is None or not golds:
                continue
            g = st.median(golds)
            cnt["under" if k < g else "exact" if k == g else "over"] += 1
        tot = sum(cnt.values())
        out.append(f"- {label}: duration stated within 25% of the gold duration {100 * within / n:.1f}%, consistent "
                   f"with its own interval {100 * consistent / n:.1f}% (n = {n}); counting under / exact / over "
                   f"{100 * cnt['under'] / tot:.0f} / {100 * cnt['exact'] / tot:.0f} / {100 * cnt['over'] / tot:.0f}%.")
    # Grounded but wrong answer vs right answer but ungrounded.
    m_all = [p[2] for p in pairs]
    out += ["", f"- Final: answer correct but IoU < 0.5 in {pct([bool(r['_ok']) and r['_iou'] < 0.5 for r in m_all]):.1f}% "
            f"of instances; IoU ≥ 0.5 but answer wrong in {pct([not r['_ok'] and r['_iou'] >= 0.5 for r in m_all]):.1f}%; "
            f"both right {pct([gq(r) for r in m_all]):.1f}%.", ""]

    # 7. Per memory
    out += ["## 7. Main failure modes per memory (final method)", "",
            "| Memory | IoU≥0.5 | too short | before gold | after gold | partial | too long | no/unparsed interval |",
            "|---|---|---|---|---|---|---|---|"]
    for name, ps in per_memory.items():
        c = collections.Counter(failure_mode(m) for _, _, m in ps)
        n = len(ps)
        f = lambda *ks: f"{100 * sum(c[k] for k in ks) / n:.1f}"
        out.append(f"| {name} | {f('IoU>=0.5, answer correct', 'IoU>=0.5, answer wrong')} | {f('inside gold, too short')} | "
                   f"{f('disjoint, before gold')} | {f('disjoint, after gold')} | {f('partial overlap')} | "
                   f"{f('covers gold, too long')} | {f('interval missing or unparsed', 'no answer line (list runs on)')} |")
    out.append("")
    # 8. Origin of the interval errors: is the final interval the span of the listed moments (so errors are born in
    # which moments get listed), or does the model change the span when writing the Time line?
    times = re.compile(r"^\s*(?:Seen|Event):\s*([\d.]+)(?:\s*-\s*([\d.]+))?", re.M)
    out += ["## 8. Origin of the interval errors (final method)", "",
            "Whether the `Time:` interval is the span of the listed moments (first to last `Seen:` time, ±1 s).", "",
            "| Outcome | instances | interval = span of listed moments | other |", "|---|---|---|---|"]
    for mode in ("IoU>=0.5, answer correct", "IoU>=0.5, answer wrong", "disjoint, before gold",
                 "inside gold, too short", "partial overlap", "disjoint, after gold"):
        sel = [m for _, _, m in pairs if failure_mode(m) == mode]
        span = 0
        for m in sel:
            ts = [float(x) for a, bb in times.findall(m["pred_raw"] or "") for x in (a, bb) if x]
            iv = interval(m)
            span += bool(ts) and iv is not None and abs(iv[0] - min(ts)) <= 1 and abs(iv[1] - max(ts)) <= 1
        out.append(f"| {mode} | {len(sel)} | {100 * span / max(len(sel), 1):.0f}% | "
                   f"{100 * (len(sel) - span) / max(len(sel), 1):.0f}% |")
    out.append("")

    # 9. Case studies (random 10%, first two instances of each failure mode in question order).
    out += ["## 9. Case studies (random 10%, final method)", ""]
    shown = collections.Counter()
    for q, b, m in per_memory["Random 10%"]:
        mode = failure_mode(m)
        if mode.startswith("IoU>=0.5, answer correct") or shown[mode] >= 2:
            continue
        shown[mode] += 1
        raw = (m["pred_raw"] or "").strip().replace("\n", " ⏎ ")
        braw = (b["pred_raw"] or "").strip().replace("\n", " ⏎ ")
        out += [f"**{mode}** — {CATS[m['question_category']]}: *{m['question']}* (question at {m['_qt']:.0f} s; gold "
                f"[{m['_gs']:.0f}, {m['_ge']:.0f}] s, \"{m['answer']}\")",
                f"- baseline (IoU {b['_iou']:.2f}): {braw[:260]}{'…' if len(braw) > 260 else ''}",
                f"- final (IoU {m['_iou']:.2f}): {raw[:260]}{'…' if len(raw) > 260 else ''}", ""]
    open("logs/phase21/error_analysis.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
