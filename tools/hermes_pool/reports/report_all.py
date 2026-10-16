"""Every table of phases 28-33 from the result directories, with paired bootstrap intervals (2,000 resamples, 95%,
* = excludes 0). Grounding: official S-EMBER-format parser, unparsed = IoU 0. Acc./GQ@0.5 (S-EMBER only) from one
judging session per table (answer_judgments_<session>.jsonl); '–' when a run has not been judged in that session.
Usage: cd /home/chieu.nguyen/HERMES && python3 /home/chieu.nguyen/hermes_pool/reports/report_all.py > report.md
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, "eval/sember")
from eval_grounding import parse_grounding_response as official_parse, temporal_iou  # noqa: E402
import re  # noqa: E402

# --mss: also read minute:second times ("Time: [1:45.0, 2:07.0]"; the model writes them on videos over a minute and
# the S-EMBER parser accepts seconds only). Applied to every run alike; the default is the official parser.
MSS = "--mss" in sys.argv


def parse_grounding_response(text):
    out = official_parse(text)
    if out[1] is None and MSS:
        text = re.sub(r"\b(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\b",  # h:mm:ss first
                      lambda m: f"{3600 * int(m.group(1)) + 60 * int(m.group(2)) + float(m.group(3)):.1f}", text)
        out = official_parse(re.sub(r"\b(\d{1,2}):(\d{1,2}(?:\.\d+)?)\b",
                                    lambda m: f"{60 * int(m.group(1)) + float(m.group(2)):.1f}", text))
    return out

RES = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/"
S = RES + "sember_grounding/uniform-n64-time-count-location-"
C, Q, A = RES + "charades_sta/", RES + "qvhighlights/", RES + "activitynet/"


def s(tag, bs8=False):
    return f"{S}{tag}-timeline{'-bs8' if bs8 else ''}-v300-native-time"


def load(path, session=None):
    if not os.path.exists(path + "/results.csv"):
        return None
    rows = {}
    for r in csv.DictReader(open(path + "/results.csv")):
        _, st, en = parse_grounding_response(r["pred_raw"])
        iou = temporal_iou(st, en, float(r["answer_start_time"]), float(r["answer_end_time"])) if st is not None else 0.0
        rows[r["question_id"]] = {"iou": iou, "r3": iou >= 0.3, "r5": iou >= 0.5, "r7": iou >= 0.7,
                                  "unparsed": st is None}
    # session: one judging session, or a list tried in order (each run is judged in exactly one of them).
    jp = next((f"{path}/answer_judgments_{x}.jsonl" for x in ([session] if isinstance(session, str) else session or [])
               if os.path.exists(f"{path}/answer_judgments_{x}.jsonl")), None)
    if jp:
        for j in map(json.loads, open(jp)):
            if j["question_id"] in rows:
                rows[j["question_id"]]["acc"] = bool(j["correct"])
                rows[j["question_id"]]["gq5"] = bool(j["correct"]) and rows[j["question_id"]]["r5"]
    return rows


def mean(rows, k):
    v = [float(r[k]) for r in rows.values() if k in r]
    return f"{100 * sum(v) / len(v):.1f}" if v and len(v) == len(rows) else "–"


def delta(a, b, k, n=2000):
    if a is None or b is None or mean(a, k) == "–" or mean(b, k) == "–" or set(a) != set(b):
        return "–"
    d = [float(b[q][k]) - float(a[q][k]) for q in a]
    rng = random.Random(0)
    bs = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    m, lo, hi = 100 * sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n)]
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"


def table(title, rows, session=None, ref=None, judged=False, keys=("iou", "r5")):
    """rows: [(label, path)]; ref: label every other row is compared with (paired)."""
    data = {label: load(path, session) for label, path in rows}
    cols = (["Acc.", "GQ@0.5"] if judged else []) + ["R@0.3", "R@0.5", "R@0.7", "mIoU", "Unparsed"]
    names = {"iou": "mIoU", "r5": "R@0.5", "acc": "Acc."}
    extra = [f"Δ{names[k]} vs {ref}" for k in keys] if ref else []
    print(f"\n### {title}\n\n| Method | n | " + " | ".join(cols + extra) + " |\n|" + "---|" * (2 + len(cols) + len(extra)))
    for label, _ in rows:
        r = data[label]
        if r is None:
            print(f"| {label} | pending |" + " |" * (len(cols) + len(extra)))
            continue
        vals = ([mean(r, "acc"), mean(r, "gq5")] if judged else []) + [mean(r, "r3"), mean(r, "r5"), mean(r, "r7"),
                                                                        mean(r, "iou"),
                                                                        str(sum(x["unparsed"] for x in r.values()))]
        if ref:
            vals += ["–" if label == ref else delta(data[ref], r, k) for k in keys]
        print(f"| {label} | {len(r)} | " + " | ".join(vals) + " |")
    return data


def pairs(title, items, session=None, keys=("iou", "r5", "acc")):
    """items: [(comparison label, path a, path b)] -> paired deltas b - a."""
    names = {"iou": "ΔmIoU", "r5": "ΔR@0.5", "acc": "ΔAcc."}
    print(f"\n### {title}\n\n| Comparison | " + " | ".join(names[k] for k in keys) + " |\n|" + "---|" * (1 + len(keys)))
    for label, a, b in items:
        A, B = load(a, session), load(b, session)
        print(f"| {label} | " + " | ".join(delta(A, B, k) if A and B else "pending" for k in keys) + " |")


print("# Results, phases 28-34 (Qwen3-VL-8B, 64 frames)" + (" — m:ss times read as seconds" if MSS else " — official parser"))

F30, F31 = "fresh30", "fresh31"
table("S-EMBER grounded QA, 10% of visual tokens (475 questions; one judging session)", [
    ("Unpruned (100%)", S + "timeline-v300-native-time"),
    ("Random", s("offline-random-keep0.1", True)),
    ("FastVID", s("p21d-fastvid10-base")), ("VidCom²", s("p22-vidcom210-base")),
    ("FlashVID", s("p22-flashvid10-base")), ("SemVID", s("p25-u64-semvid10")),
    ("Cover (question top-k, uniform frames)", s("p24-u64-qtopk10")),
    ("Retrieve + random", s("p24-rc64-rand10")), ("Retrieve + Cover", s("p25-rc64-qtopk10")),
    ("Random + PCD", s("random10-cd-blind-adapt1.0-scopetime", True)),
    ("Random + PCD + reserve", s("p31-random10-pcd-ar64")),
    ("SemVID + PCD", s("p25-u64-semvid10-pcd")),
    ("SemVID + PCD + reserve (not judged in this session)", s("p35-u64-semvid10-pcd-ar64")),
    ("Retrieve + Cover + PCD", s("p25-rc64-qtopk10-pcd")),
    ("Retrieve + Cover + PCD + reserve (final)", s("p31-rc64-qtopk10-pcd-ar64")),
    ("final + coverage gate", s("p31-rc64g-qtopk10-pcd-ar64")),
], session=F31, ref="Random", judged=True, keys=("iou", "r5", "acc"))

pairs("S-EMBER: effect of each fix (paired)", [
    ("answer reserve: Random + PCD", s("random10-cd-blind-adapt1.0-scopetime", True), s("p31-random10-pcd-ar64")),
    ("answer reserve: Retrieve + Cover + PCD", s("p25-rc64-qtopk10-pcd"), s("p31-rc64-qtopk10-pcd-ar64")),
    ("coverage gate: Retrieve + Cover", s("p25-rc64-qtopk10"), s("p31-rc64g-qtopk10")),
    ("coverage gate: final", s("p31-rc64-qtopk10-pcd-ar64"), s("p31-rc64g-qtopk10-pcd-ar64")),
    ("final vs SemVID + PCD", s("p25-u64-semvid10-pcd"), s("p31-rc64-qtopk10-pcd-ar64")),
    ("final vs SemVID + PCD + reserve (same decoding)", s("p35-u64-semvid10-pcd-ar64"), s("p31-rc64-qtopk10-pcd-ar64")),
    ("answer reserve: SemVID + PCD", s("p25-u64-semvid10-pcd"), s("p35-u64-semvid10-pcd-ar64")),
    ("final vs Random + PCD + reserve", s("p31-random10-pcd-ar64"), s("p31-rc64-qtopk10-pcd-ar64")),
], session=F31)

budget_rows = []
for k, tag in ((5, "5"), (10, "10"), (20, "20")):
    if k == 10:
        budget_rows += [("Random 10%", s("offline-random-keep0.1", True)), ("SemVID 10%", s("p25-u64-semvid10")),
                        ("Retrieve + Cover 10%", s("p25-rc64-qtopk10")),
                        ("Random + PCD 10%", s("random10-cd-blind-adapt1.0-scopetime", True)),
                        ("SemVID + PCD 10%", s("p25-u64-semvid10-pcd")),
                        ("Retrieve + Cover + PCD 10%", s("p25-rc64-qtopk10-pcd")),
                        ("Random + PCD + reserve 10%", s("p31-random10-pcd-ar64")),
                        ("SemVID + PCD + reserve 10%", s("p35-u64-semvid10-pcd-ar64")),
                        ("final 10%", s("p31-rc64-qtopk10-pcd-ar64"))]
    else:
        budget_rows += [(f"Random {k}%", s(f"p30-random{tag}")), (f"SemVID {k}%", s(f"p30-u64-semvid{tag}")),
                        (f"Retrieve + Cover {k}%", s(f"p30-rc64-qtopk{tag}")),
                        (f"Random + PCD {k}%", s(f"p30-random{tag}-pcd")),
                        (f"SemVID + PCD {k}%", s(f"p30-u64-semvid{tag}-pcd")),
                        (f"Retrieve + Cover + PCD {k}%", s(f"p30-rc64-qtopk{tag}-pcd")),
                        (f"Random + PCD + reserve {k}%", s(f"p32-random{tag}-pcd-ar64")),
                        (f"SemVID + PCD + reserve {k}%", s(f"p35-u64-semvid{tag}-pcd-ar64")),
                        (f"final {k}%", s(f"p32-rc64-qtopk{tag}-pcd-ar64"))]
table("S-EMBER token budget (phase 30 / 32); Acc. from session fresh30 (p32 rows: fresh32)", budget_rows,
      session=[F30, "fresh32"], judged=True)
pairs("S-EMBER budget: method gaps at each budget (paired)", [
    (f"{k}%: {b} vs {a}", s(pa), s(pb)) for k, _, a, b, pa, pb in [
        (5, "", "SemVID", "Retrieve + Cover", "p30-u64-semvid5", "p30-rc64-qtopk5"),
        (20, "", "SemVID", "Retrieve + Cover", "p30-u64-semvid20", "p30-rc64-qtopk20"),
        (5, "", "Random", "Retrieve + Cover", "p30-random5", "p30-rc64-qtopk5"),
        (20, "", "Random", "Retrieve + Cover", "p30-random20", "p30-rc64-qtopk20"),
        (5, "", "SemVID + PCD", "Retrieve + Cover + PCD", "p30-u64-semvid5-pcd", "p30-rc64-qtopk5-pcd"),
        (20, "", "SemVID + PCD", "Retrieve + Cover + PCD", "p30-u64-semvid20-pcd", "p30-rc64-qtopk20-pcd"),
        (5, "", "Random + PCD + reserve", "final", "p32-random5-pcd-ar64", "p32-rc64-qtopk5-pcd-ar64"),
        (20, "", "Random + PCD + reserve", "final", "p32-random20-pcd-ar64", "p32-rc64-qtopk20-pcd-ar64"),
        (5, "", "SemVID + PCD + reserve", "final", "p35-u64-semvid5-pcd-ar64", "p32-rc64-qtopk5-pcd-ar64"),
        (10, "", "SemVID + PCD + reserve", "final", "p35-u64-semvid10-pcd-ar64", "p31-rc64-qtopk10-pcd-ar64"),
        (20, "", "SemVID + PCD + reserve", "final", "p35-u64-semvid20-pcd-ar64", "p32-rc64-qtopk20-pcd-ar64")]],
    session=F30, keys=("iou", "r5"))

F32 = "fresh32"
table("S-EMBER: per-layer vs shared token set (phase 32; Acc. from session fresh32)", [
    ("Random, per layer", s("offline-random-keep0.1", True)), ("Random, shared", s("p32-random10-shared")),
    ("Random + PCD + reserve, per layer", s("p31-random10-pcd-ar64")),
    ("Random + PCD + reserve, shared", s("p32-random10-shared-pcd-ar64")),
    ("Retrieve + Cover, per layer", s("p25-rc64-qtopk10")), ("Retrieve + Cover, shared", s("p32-rc64-qtopk10-shared")),
    ("final, per layer", s("p31-rc64-qtopk10-pcd-ar64")), ("final, shared", s("p32-rc64-qtopk10-shared-pcd-ar64")),
], session=F32, judged=True)
pairs("S-EMBER: shared minus per-layer (paired)", [
    ("Random", s("offline-random-keep0.1", True), s("p32-random10-shared")),
    ("Random + PCD + reserve", s("p31-random10-pcd-ar64"), s("p32-random10-shared-pcd-ar64")),
    ("Retrieve + Cover", s("p25-rc64-qtopk10"), s("p32-rc64-qtopk10-shared")),
    ("final", s("p31-rc64-qtopk10-pcd-ar64"), s("p32-rc64-qtopk10-shared-pcd-ar64")),
], session=F32)
seeds = table("S-EMBER: random selection seeds (phase 32)", [
    ("Random, seed 0", s("offline-random-keep0.1", True)), ("Random, seed 1", s("p32-random10-seed1")),
    ("Random, seed 2", s("p32-random10-seed2"))], session=F32, judged=True)
vals = [float(mean(r, "iou")) for r in seeds.values() if r is not None]
if len(vals) > 1:
    m = sum(vals) / len(vals)
    print(f"\nRandom mIoU over {len(vals)} seeds: {m:.1f} ± {(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5:.1f} (sd)")

for name, base, extra in (("Charades-STA (500 test queries, ~30 s videos)", C, [
        ("Random + PCD + reserve", "p31-random10-pcd-ar64"), ("SemVID + PCD + reserve", "p31-semvid10-pcd-ar64"),
        ("Retrieve + Cover + PCD + reserve", "p31-rc64-qtopk10-pcd-ar64"),
        ("Retrieve(gated) + Cover", "p31-rc64g-qtopk10"),
        ("final + coverage gate", "p31-rc64g-qtopk10-pcd-ar64")]),
        ("QVHighlights (500 single-window val queries, ~150 s videos)", Q, [
            ("Retrieve + random", "rc64-rand10"), ("Random + PCD", "random10-pcd"),
            ("Random + PCD + reserve", "random10-pcd-ar64"), ("SemVID + PCD + reserve", "semvid10-pcd-ar64"),
            ("Retrieve + Cover + PCD", "rc64-qtopk10-pcd"), ("final", "rc64-qtopk10-pcd-ar64"),
            ("final, 5% of tokens", "rc64-qtopk5-pcd-ar64"), ("final, 20% of tokens", "rc64-qtopk20-pcd-ar64")]),
        ("ActivityNet Captions (500 val_1 queries, one per video; 224 px / 3 fps copy, so 10% = only 120-300 tokens: "
         "an extreme-budget stress test — see the 30% / 50% table)", A, [
            ("Retrieve + random", "rc64-rand10"), ("Random + PCD + reserve", "random10-pcd-ar64"),
            ("SemVID + PCD + reserve", "semvid10-pcd-ar64"), ("final", "rc64-qtopk10-pcd-ar64")])):
    rows = [("Unpruned (100%)", base + ("unpruned-bs1" if base == C else "unpruned")), ("Random", base + "random10"),
            ("FastVID", base + "fastvid10"), ("VidCom²", base + "vidcom210"), ("FlashVID", base + "flashvid10"),
            ("SemVID", base + "semvid10"), ("Cover (uniform frames)", base + "u64-qtopk10"),
            ("Retrieve + Cover", base + "rc64-qtopk10")]
    if base == A:  # FastVID and VidCom² were the lowest on the other three datasets and are not run here
        rows = [r for r in rows if r[0] not in ("FastVID", "VidCom²")]
    if base == C:
        rows += [("Random + PCD", C + "random10-pcd"), ("SemVID + PCD", C + "semvid10-pcd"),
                 ("Retrieve + Cover + PCD", C + "rc64-qtopk10-pcd")]
    rows += [(label, base + tag) for label, tag in extra]
    table(name + ", 10% of visual tokens", rows, ref="Random")
    if base == C:
        pairs("Charades: effect of each fix (paired)", [
            ("answer reserve: Random + PCD", C + "random10-pcd", C + "p31-random10-pcd-ar64"),
            ("answer reserve: SemVID + PCD", C + "semvid10-pcd", C + "p31-semvid10-pcd-ar64"),
            ("answer reserve: Retrieve + Cover + PCD", C + "rc64-qtopk10-pcd", C + "p31-rc64-qtopk10-pcd-ar64"),
            ("coverage gate: Retrieve + Cover", C + "rc64-qtopk10", C + "p31-rc64g-qtopk10"),
            ("final+gate vs Random + PCD + reserve", C + "p31-random10-pcd-ar64", C + "p31-rc64g-qtopk10-pcd-ar64"),
        ], keys=("iou", "r5"))
    elif base == A:
        pairs("ActivityNet Captions: key comparisons (paired)", [
            ("final vs Random + PCD + reserve", A + "random10-pcd-ar64", A + "rc64-qtopk10-pcd-ar64"),
            ("final vs SemVID + PCD + reserve", A + "semvid10-pcd-ar64", A + "rc64-qtopk10-pcd-ar64"),
            ("final vs Unpruned", A + "unpruned", A + "rc64-qtopk10-pcd-ar64"),
            ("Retrieve + Cover vs Random", A + "random10", A + "rc64-qtopk10"),
            ("Cover vs Random", A + "random10", A + "u64-qtopk10"),
            ("PCD + reserve on Random", A + "random10", A + "random10-pcd-ar64"),
        ], keys=("iou", "r5"))
    else:
        pairs("QVHighlights: key comparisons (paired)", [
            ("answer reserve: Random + PCD", Q + "random10-pcd", Q + "random10-pcd-ar64"),
            ("answer reserve: Retrieve + Cover + PCD", Q + "rc64-qtopk10-pcd", Q + "rc64-qtopk10-pcd-ar64"),
            ("final vs Unpruned", Q + "unpruned", Q + "rc64-qtopk10-pcd-ar64"),
            ("final vs Random + PCD + reserve", Q + "random10-pcd-ar64", Q + "rc64-qtopk10-pcd-ar64"),
            ("final vs SemVID + PCD + reserve", Q + "semvid10-pcd-ar64", Q + "rc64-qtopk10-pcd-ar64"),
            ("Retrieve + Cover vs Random", Q + "random10", Q + "rc64-qtopk10"),
            ("Retrieve + random vs Random", Q + "random10", Q + "rc64-rand10"),
            ("PCD + reserve on Random", Q + "random10", Q + "random10-pcd-ar64"),
        ], keys=("iou", "r5"))

budget_anet = []
for k in ("30", "50"):
    budget_anet += [(f"Random {k}%", A + f"random{k}"), (f"SemVID {k}%", A + f"semvid{k}"),
                    (f"Cover (uniform frames) {k}%", A + f"u64-qtopk{k}"), (f"Retrieve + Cover {k}%", A + f"rc64-qtopk{k}"),
                    (f"Random + PCD + reserve {k}%", A + f"random{k}-pcd-ar64"),
                    (f"SemVID + PCD + reserve {k}%", A + f"semvid{k}-pcd-ar64"), (f"final {k}%", A + f"rc64-qtopk{k}-pcd-ar64")]
table("ActivityNet Captions at 30% and 50% of the visual tokens (main ActivityNet comparison)",
      [("Unpruned (100%)", A + "unpruned")] + budget_anet)
pairs("ActivityNet at 30% / 50%: key comparisons (paired)", [
    (f"{k}%: {b} vs {a}", A + pa, A + pb) for k in ("30", "50") for a, b, pa, pb in [
        ("Random", "Retrieve + Cover", f"random{k}", f"rc64-qtopk{k}"),
        ("Random", "Cover", f"random{k}", f"u64-qtopk{k}"),
        ("SemVID", "Retrieve + Cover", f"semvid{k}", f"rc64-qtopk{k}"),
        ("Random + PCD + reserve", "final", f"random{k}-pcd-ar64", f"rc64-qtopk{k}-pcd-ar64"),
        ("SemVID + PCD + reserve", "final", f"semvid{k}-pcd-ar64", f"rc64-qtopk{k}-pcd-ar64"),
        ("Unpruned", "final", "unpruned", f"rc64-qtopk{k}-pcd-ar64")]], keys=("iou", "r5"))

# Phase 36: coverage vs per-frame evidence at a fixed token budget (F frames at ratio r * 64 / F).
table("Charades-STA: 64 vs 32 frames at the same token budget (64 frames at 10% = 32 frames at 20%)", [
    ("Unpruned, 64 frames", C + "unpruned-bs1"), ("Unpruned, 32 frames", C + "f32-unpruned"),
    ("Random, 64", C + "random10"), ("Random, 32", C + "f32-random20"),
    ("SemVID, 64", C + "semvid10"), ("SemVID, 32", C + "f32-semvid20"),
    ("Cover, 64", C + "u64-qtopk10"), ("Cover, 32", C + "f32-u32-qtopk20"),
    ("Random + PCD + reserve, 64", C + "p31-random10-pcd-ar64"), ("Random + PCD + reserve, 32", C + "f32-random20-pcd-ar64"),
    ("SemVID + PCD + reserve, 64", C + "p31-semvid10-pcd-ar64"), ("SemVID + PCD + reserve, 32", C + "f32-semvid20-pcd-ar64"),
    ("Cover + PCD + reserve, 64 (= gated final)", C + "p31-rc64g-qtopk10-pcd-ar64"),
    ("Cover + PCD + reserve, 32", C + "f32-u32-qtopk20-pcd-ar64")])
pairs("Charades at 32 frames: paired comparisons", [
    ("32 minus 64 frames: Random + PCD + reserve", C + "p31-random10-pcd-ar64", C + "f32-random20-pcd-ar64"),
    ("32 minus 64 frames: Cover + PCD + reserve", C + "p31-rc64g-qtopk10-pcd-ar64", C + "f32-u32-qtopk20-pcd-ar64"),
    ("32 frames: Cover + PCD + reserve vs Random + PCD + reserve", C + "f32-random20-pcd-ar64", C + "f32-u32-qtopk20-pcd-ar64"),
    ("32 frames: Cover + PCD + reserve vs SemVID + PCD + reserve", C + "f32-semvid20-pcd-ar64", C + "f32-u32-qtopk20-pcd-ar64"),
    ("32 frames: Cover + PCD + reserve vs Unpruned", C + "f32-unpruned", C + "f32-u32-qtopk20-pcd-ar64"),
], keys=("iou", "r5"))
for name, base, budget, rows64 in (
        ("QVHighlights at ~260 tokens (64 frames at 5%)", Q, ("10", "20"), [("final, 64 frames (5%)", "rc64-qtopk5-pcd-ar64")]),
        ("ActivityNet at ~120-300 tokens (64 frames at 10%)", A, ("20", "40"), [
            ("Random + PCD + reserve, 64 frames (10%)", "random10-pcd-ar64"),
            ("SemVID + PCD + reserve, 64 frames (10%)", "semvid10-pcd-ar64"), ("final, 64 frames (10%)", "rc64-qtopk10-pcd-ar64")])):
    k32, k16 = budget
    rows = [(label, base + tag) for label, tag in rows64]
    for n, k in ((32, k32), (16, k16)):
        rows += [(f"Random + PCD + reserve, {n} frames ({k}%)", base + f"f{n}-random{k}-pcd-ar64"),
                 (f"SemVID + PCD + reserve, {n} frames ({k}%)", base + f"f{n}-semvid{k}-pcd-ar64"),
                 (f"final, {n} frames ({k}%)", base + f"f{n}-rc{n}-qtopk{k}-pcd-ar64")]
    table(name + ": same token budget, 64 / 32 / 16 frames", rows)
    pairs(name + ": paired comparisons", [
        (f"{n} frames: final vs Random + PCD + reserve", base + f"f{n}-random{k}-pcd-ar64", base + f"f{n}-rc{n}-qtopk{k}-pcd-ar64")
        for n, k in ((32, k32), (16, k16))] + [
        (f"{n} frames: final vs SemVID + PCD + reserve", base + f"f{n}-semvid{k}-pcd-ar64", base + f"f{n}-rc{n}-qtopk{k}-pcd-ar64")
        for n, k in ((32, k32), (16, k16))] + [
        (f"final: {n} minus 64 frames", base + rows64[-1][1], base + f"f{n}-rc{n}-qtopk{k}-pcd-ar64")
        for n, k in ((32, k32), (16, k16))], keys=("iou", "r5"))
