"""NExT-GQA joint protocol: one reply gives the time span and the choice (LOVER, arXiv 2609.15224, Appendix A.2,
Figure 6: inference prompt without timestamp rendering, used for their Qwen3-VL / Time-R1 baselines).
Same 1,005 test questions as prepare.py, one sample per question: <vid>_<qid>_joint.
Adapted for NExT-GQA's five options: "from A, B, C, or D" -> "from A, B, C, D, or E"; options listed as "A. ..."
under the question, as in LOVER's Figure 1. Everything else verbatim.
Usage: python3 prepare_joint.py   (after prepare.py)
"""
import json

D = "/nfs-stor/chieu.nguyen/nextgqa"
REPO = "/home/chieu.nguyen/hermes_pool/nextgqa/repo/datasets/nextgqa"
LOVER = ("You are given a video and a multiple-choice question.\n"
         "Question:{}\n"
         "Task:\n"
         "1. First, locate the time period in the video that is most relevant to answering the question.\n"
         "2. Then, select the correct answer from A, B, C, D, or E.\n"
         "Output format:\n"
         "- Output your thought process within the <think> </think> tags, including analysis with either specific "
         "time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n"
         "- Then, provide the start and end times (in seconds, precise to two decimal places) in the format "
         "\"start time to end time\" within the <answer> </answer> tags.\n"
         "- Put your final choice inside <choice> </choice>.\n"
         "Example:\n"
         "<think>... </think>\n"
         "<answer>12.54 to 17.83</answer>\n"
         "<choice>B</choice>")


def main():
    import csv
    rows = {(r["video_id"], r["qid"]): r for r in csv.DictReader(open(f"{REPO}/test.csv"))}
    out = []
    for line in open(f"{D}/nextgqa_test_subset.jsonl"):
        s = json.loads(line)
        if not s["question_id"].endswith("_ans"):
            continue
        m = s["nextgqa"]
        r = rows[(m["vid"], m["qid"])]
        cap = lambda x: x[0].upper() + x[1:] if x else x  # noqa: E731  (as prepare.py)
        q = cap(r["question"]) + "?" + "".join(f"\n{chr(ord('A') + i)}. {cap(r[f'a{i}'])}" for i in range(5))
        out.append({**s, "question_id": s["question_id"][:-4] + "_joint", "question": LOVER.format(q)})
    with open(f"{D}/nextgqa_test_joint.jsonl", "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print(f"{len(out)} joint samples")
    print(out[0]["question"])


if __name__ == "__main__":
    main()
