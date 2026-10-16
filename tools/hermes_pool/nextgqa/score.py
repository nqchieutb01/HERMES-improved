"""Score our NExT-GQA runs with the official evaluation (doc-doc/NExT-GQA code/TempGQA/eval_ground.py, unmodified).

Answers: VideoMind's check_ans (first token of the reply, brackets and dots stripped, must be A-E). Grounding: the
first time span TimeLens' extract_time finds in the grounding reply; none found -> [-100, -100] (IoU = IoP = 0).
Official metrics on the subset: Acc&GQA (correct answer and IoP >= 0.5), mIoP, IoP@0.3/0.5, mIoU, IoU@0.3/0.5;
plus Acc@QA and Acc@QA / Acc&GQA per question type (same rules).
Joint protocol (--joint; prepare_joint.py, LOVER's prompt): one reply per question; answer = the letter in the last
<choice> </choice>, span = the first two numbers in the last <answer> </answer>; either missing -> unanswered /
[-100, -100], counted in "unanswered" / "ungrounded".
Open-ended (--open oe|tl; prepare_open.py, judge_open.py): answer correct = the judge's "yes" in
<run dir>/open_judgments.jsonl (TOGA's Open-Acc); span = TimeLens' extract_time on the _grd reply (oe) or S-EMBER's own
parser on the Time: line of the _tl reply (tl). Open-Acc&GQA = judged correct and IoP >= 0.5 (official eval_ground).
Usage: python score.py [--partial] [--joint] [--open oe|tl] <run dir with results.csv> [...]   (--partial: chunk CSVs <n>_<i>.csv of a run in
progress, scored on the questions whose two samples are both done)   (hermes-qwen python: the official util imports torch)
"""
import contextlib
import csv
import io
import json
import re
import sys

REPO = "/home/chieu.nguyen/hermes_pool/nextgqa/repo"
SUBSET = "/nfs-stor/chieu.nguyen/nextgqa/nextgqa_test_subset.jsonl"
sys.path.insert(0, f"{REPO}/code/TempGQA")
sys.path.insert(0, "/home/chieu.nguyen/hermes_pool/timelens/repo")
from eval_ground import eval_ground, get_tIoU  # noqa: E402
from timelens.utils import extract_time  # noqa: E402


def check_ans(response, n_options=5):
    """VideoMind videomind/eval/eval_auto.py check_ans: the letter the reply starts with (None if not A-E)."""
    b = response.lower().split(" ")[0].replace("(", "").replace(")", "").replace(".", "") if response.strip() else ""
    b = b[:1]
    return b.upper() if b in [chr(ord("a") + i) for i in range(n_options)] else None


def parse_choice(reply):
    m = re.findall(r"<choice>\s*\(?([A-Ea-e])\b", reply)
    return m[-1].upper() if m else None


def parse_span(reply):
    m = re.findall(r"<answer>(.*?)</answer>", reply, flags=re.S)
    nums = re.findall(r"\d+(?:\.\d+)?", m[-1]) if m else []
    return sorted(float(x) for x in nums[:2]) if len(nums) >= 2 else None


def score_open(run_dir, proto):
    """Open-ended scoring with the judge's verdicts; the multiple-choice path below is unchanged."""
    sys.path.insert(0, "/home/chieu.nguyen/HERMES/eval/sember")
    from eval_grounding import parse_grounding_response
    preds = {r["question_id"]: r["pred_raw"] for r in csv.DictReader(open(f"{run_dir}/results.csv"))}
    verdict = {e["key"]: e["verdict"] for e in map(json.loads, open(f"{run_dir}/open_judgments.jsonl"))}
    gt_ground, pred_ground, pred_qa, per_q, clock = {}, {}, {}, {}, 0
    for line in open(SUBSET):
        s = json.loads(line)
        if not s["question_id"].endswith("_ans"):
            continue
        m = s["nextgqa"]
        key, base = f"{m['vid']}_{m['qid']}", s["question_id"][:-4]
        gt_ground.setdefault(m["vid"], {"duration": None, "location": {}})["location"][m["qid"]] = m["spans"]
        pred_qa[key] = {"answer": "yes", "prediction": verdict.get(base, "no")}
        if proto == "oe":
            ts = extract_time(preds.get(f"{base}_grd", ""))
            span = list(ts[0]) if ts else None
        else:
            reply = preds.get(f"{base}_tl", "")
            _, st, en = parse_grounding_response(reply)
            span = [st, en] if st is not None and en is not None else None
            clock += bool(re.search(r"Time:[^\n]*\d+:\d\d", reply))
        pred_ground[key] = span if span else [-100.0, -100.0]
        per_q[key] = m["type"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        eval_ground(gt_ground, pred_ground, pred_qa)
    official = out.getvalue().strip().splitlines()[-1].split()
    names = ["Open-Acc&GQA", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5"]
    res = {"n": len(pred_qa), "Open-Acc@QA": round(100 * sum(v["prediction"] == "yes" for v in pred_qa.values())
                                                 / len(pred_qa), 1),
           **dict(zip(names, map(float, official))),
           "judge_unparsed": sum(v["prediction"] == "unparsed" for v in pred_qa.values()),
           "ungrounded": sum(p == [-100.0, -100.0] for p in pred_ground.values()), "clock_time_lines": clock}
    by_type = {}
    for key, t in per_q.items():
        vid, qid = key.split("_")
        iop = max(get_tIoU(loc, pred_ground[key])[1] for loc in gt_ground[vid]["location"][qid])
        ok = pred_qa[key]["prediction"] == "yes"
        st = by_type.setdefault(t, [0, 0, 0])
        st[0] += 1
        st[1] += ok
        st[2] += ok and iop >= 0.5
    res["by_type"] = {t: {"n": n, "Open-Acc@QA": round(100 * a / n, 1), "Open-Acc&GQA": round(100 * g / n, 1)}
                      for t, (n, a, g) in sorted(by_type.items())}
    json.dump({"pred_ground": pred_ground, "pred_qa": pred_qa, "metrics": res},
              open(f"{run_dir}/nextgqa_open_eval.json", "w"))
    return res


def score(run_dir, partial=False, joint=False):
    import glob
    files = sorted(glob.glob(f"{run_dir}/*_*.csv")) if partial else [f"{run_dir}/results.csv"]
    preds = {r["question_id"]: r["pred_raw"] for f in files for r in csv.DictReader(open(f))}
    meta = {}
    for line in open(SUBSET):
        s = json.loads(line)
        base = s["question_id"].rsplit("_", 1)[0]
        done = f"{base}_joint" in preds if joint else (f"{base}_ans" in preds and f"{base}_grd" in preds)
        if not partial or done:
            meta[s["question_id"]] = s["nextgqa"]
    gt_ground, pred_ground, pred_qa, per_q = {}, {}, {}, {}
    for qid_full, m in meta.items():
        key = f"{m['vid']}_{m['qid']}"
        gt_ground.setdefault(m["vid"], {"duration": None, "location": {}})["location"][m["qid"]] = m["spans"]
        reply = preds.get(qid_full.rsplit("_", 1)[0] + "_joint", "") if joint else preds.get(qid_full, "")
        if qid_full.endswith("_ans"):
            pred_qa[key] = {"answer": m["ans"], "prediction": parse_choice(reply) if joint else check_ans(reply)}
        elif joint:
            pred_ground[key] = parse_span(reply) or [-100.0, -100.0]
        else:
            ts = extract_time(reply)
            pred_ground[key] = list(ts[0]) if ts else [-100.0, -100.0]
        per_q[key] = m["type"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        eval_ground(gt_ground, pred_ground, pred_qa)
    official = out.getvalue().strip().splitlines()[-1].split()
    acc_qa = 100 * sum(v["answer"] == v["prediction"] for v in pred_qa.values()) / len(pred_qa)
    by_type = {}
    for key, t in per_q.items():
        vid, qid = key.split("_")
        iop = max(get_tIoU(loc, pred_ground[key])[1] for loc in gt_ground[vid]["location"][qid])
        ok = pred_qa[key]["answer"] == pred_qa[key]["prediction"]
        st = by_type.setdefault(t, [0, 0, 0])
        st[0] += 1
        st[1] += ok
        st[2] += ok and iop >= 0.5
    names = ["Acc&GQA", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5"]
    res = {"n": len(pred_qa), "Acc@QA": round(acc_qa, 1), **dict(zip(names, map(float, official))),
           "unanswered": sum(v["prediction"] is None for v in pred_qa.values()),
           "ungrounded": sum(p == [-100.0, -100.0] for p in pred_ground.values()),
           "by_type": {t: {"n": n, "Acc@QA": round(100 * a / n, 1), "Acc&GQA": round(100 * g / n, 1)}
                       for t, (n, a, g) in sorted(by_type.items())}}
    if not partial:
        json.dump({"pred_ground": pred_ground, "pred_qa": pred_qa, "metrics": res},
                  open(f"{run_dir}/nextgqa_eval.json", "w"))
    return res


if __name__ == "__main__":
    partial, joint = "--partial" in sys.argv, "--joint" in sys.argv
    argv = sys.argv[1:]
    proto = argv[argv.index("--open") + 1] if "--open" in argv else None
    dirs = [a for i, a in enumerate(argv) if a not in ("--partial", "--joint", "--open") and (i == 0 or argv[i - 1] != "--open")]
    for run_dir in dirs:
        r = score_open(run_dir.rstrip("/"), proto) if proto else score(run_dir.rstrip("/"), partial, joint)
        print(run_dir.rstrip("/").split("/")[-1], json.dumps({k: v for k, v in r.items() if k != "by_type"}))
        print("   by type:", json.dumps(r["by_type"]))
