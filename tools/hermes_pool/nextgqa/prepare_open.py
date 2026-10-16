"""NExT-GQA open-ended: the same 1,005 test questions as prepare.py, without the answer options.

Two protocols (one JSONL each):
- nextgqa_test_oe.jsonl (two queries, grounding_prompt=raw): <vid>_<qid>_oe = the question + LLaVA's short-answer
  instruction "Answer the question using a single word or phrase." (free-form answer), and <vid>_<qid>_grd = the
  existing TimeLens grounding query (copied from prepare.py's subset). As the multiple-choice protocol, minus options.
- nextgqa_test_tl.jsonl (joint, grounding_prompt=timeline): <vid>_<qid>_tl = the plain question; the pipeline wraps it
  in the S-EMBER timeline prompt, so one reply carries "Answer: ..." and "Time: [s, e]" (as on S-EMBER).
Answers are judged against the ground-truth answer text (judge_open.py, Video-ChatGPT / TOGA yes-no protocol).
Usage: python3 prepare_open.py   (after prepare.py)
"""
import csv
import json

D = "/nfs-stor/chieu.nguyen/nextgqa"
REPO = "/home/chieu.nguyen/hermes_pool/nextgqa/repo/datasets/nextgqa"
SHORT = "Answer the question using a single word or phrase."


def cap(s):
    return s[0].upper() + s[1:] if s else s


def main():
    rows = {(r["video_id"], r["qid"]): r for r in csv.DictReader(open(f"{REPO}/test.csv"))}
    oe, tl = [], []
    for line in open(f"{D}/nextgqa_test_subset.jsonl"):
        s = json.loads(line)
        m = s["nextgqa"]
        r = rows[(m["vid"], m["qid"])]
        q = cap(r["question"]) + "?"
        meta = {**m, "answer_text": cap(r["answer"]), "question_text": q}
        base = s["question_id"].rsplit("_", 1)[0]
        if s["question_id"].endswith("_grd"):
            oe.append({**s, "nextgqa": meta})
            continue
        common = {**s, "nextgqa": meta, "answer": cap(r["answer"])}
        oe.append({**common, "question_id": f"{base}_oe", "question": f"{q}\n{SHORT}"})
        tl.append({**common, "question_id": f"{base}_tl", "question": q})
    for name, items in (("oe", oe), ("tl", tl)):
        with open(f"{D}/nextgqa_test_{name}.jsonl", "w") as f:
            for s in items:
                f.write(json.dumps(s) + "\n")
        print(f"{name}: {len(items)} samples; e.g. {items[0]['question_id']}: {items[0]['question']!r} -> {items[0]['answer']!r}")


if __name__ == "__main__":
    main()
