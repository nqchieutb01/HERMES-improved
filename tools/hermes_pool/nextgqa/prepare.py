"""NExT-GQA (doc-doc/NExT-GQA, CVPR 2024) test subset in our grounding JSONL, two samples per question.

Subset: test videos drawn at random (seed 0) until the questions of the drawn videos reach 1,000; all questions of a
drawn video are kept (question types stay close to the full test set). Videos: NExT-QA videos fetched one by one from
the jinyoungkim/NExT-GQA Hugging Face mirror (NExTVideo/<folder>/<vidor id>.mp4 via map_vid_vidorID.json).
Per question two samples, both prompts taken from prior work (grounding_prompt=raw uses the question field verbatim):
- <vid>_<qid>_ans: VideoMind's multiple-choice answer prompt (yeliudev/VideoMind, videomind/eval/infer_auto.py):
  "<Question>?\\nOptions:\\n(A) ...\\n...\\nPlease only give the best option."
- <vid>_<qid>_grd: TimeLens' grounding prompt (TencentARC/TimeLens GROUNDER_PROMPT) with the question as the query,
  as VideoMind grounds with the question.
Usage: python3 prepare.py
"""
import csv
import json
import os
import random
import subprocess

REPO = "/home/chieu.nguyen/hermes_pool/nextgqa/repo/datasets/nextgqa"
D = "/nfs-stor/chieu.nguyen/nextgqa"
HF = "https://huggingface.co/datasets/jinyoungkim/NExT-GQA/resolve/main/NExTVideo"
N_QUESTIONS = 1000
TIMELENS = ("Please find the visual event described by the sentence '{}', determining its starting and ending times. "
            "The format should be: 'The event happens in <start time> - <end time> seconds'.")


def cap(s):
    return s[0].upper() + s[1:] if s else s


def main():
    rows = list(csv.DictReader(open(f"{REPO}/test.csv")))
    spans = json.load(open(f"{REPO}/gsub_test.json"))
    vmap = json.load(open(f"{REPO}/map_vid_vidorID.json"))
    by_video = {}
    for r in rows:
        if r["video_id"] in spans and r["qid"] in spans[r["video_id"]]["location"]:
            by_video.setdefault(r["video_id"], []).append(r)
    vids = sorted(by_video)
    random.Random(0).shuffle(vids)
    chosen, n = [], 0
    for v in vids:
        if n >= N_QUESTIONS:
            break
        chosen.append(v)
        n += len(by_video[v])
    os.makedirs(f"{D}/videos", exist_ok=True)
    token = open("/scratch/chieu.nguyen/cache/huggingface/token").read().strip()
    for v in chosen:
        out = f"{D}/videos/{v}.mp4"
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            subprocess.run(["curl", "-sSL", "--retry", "5", "-H", f"Authorization: Bearer {token}", "-o", out,
                            f"{HF}/{vmap[v]}.mp4"], check=True)
    samples, types = [], {}
    for v in chosen:
        dur = float(spans[v]["duration"])
        for r in by_video[v]:
            question = cap(r["question"]) + "?"
            options = [cap(r[f"a{i}"]) for i in range(5)]
            ans = chr(ord("A") + options.index(cap(r["answer"])))
            mcq = question + "\nOptions:" + "".join(f"\n({chr(ord('A') + i)}) {o}" for i, o in enumerate(options))
            mcq += "\nPlease only give the best option."
            gt = spans[v]["location"][r["qid"]]
            base = {"video_id": v, "question_time": dur, "duration": dur, "question_category": "moment_retrieval",
                    # Loader fields only (validated, not scored): a few official spans start at -0.1 s.
                    "answer_start_time": max(0.0, float(gt[0][0])), "answer_end_time": min(dur, float(gt[0][1])),
                    "nextgqa": {"vid": v, "qid": r["qid"], "type": r["type"], "ans": ans, "spans": gt}}
            samples.append({**base, "question_id": f"{v}_{r['qid']}_ans", "question": mcq, "answer": ans})
            samples.append({**base, "question_id": f"{v}_{r['qid']}_grd", "question": TIMELENS.format(question),
                            "answer": r["answer"]})
            types[r["type"]] = types.get(r["type"], 0) + 1
    with open(f"{D}/nextgqa_test_subset.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"{len(samples) // 2} questions on {len(chosen)} videos; types {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
