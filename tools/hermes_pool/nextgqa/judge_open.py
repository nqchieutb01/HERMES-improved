"""LLM judge for open-ended NExT-GQA answers (TOGA's Open-Acc protocol: arXiv 2506.09445, Supp. 2.2).

TOGA asks GPT-3.5 whether the prediction matches the ground-truth answer in meaning, given the question and without the
options, following the GPT-assisted evaluation of Video-ChatGPT. We use Video-ChatGPT's prompt verbatim
(mbzuai-oryx/Video-ChatGPT, quantitative_evaluation/evaluate_activitynet_qa.py: system + user message, reply
{'pred': 'yes'|'no', 'score': 0-5}) with a local judge model instead of GPT-3.5: Qwen3.8-27B-FP8 (the S-EMBER judge,
logs/phase11/judge_grounding.py), greedy, thinking off.

Jobs file, one per line: <mode> <run dir or -> <output jsonl>
  oe   answer = the reply to <vid>_<qid>_oe (two-query protocol, prepare_open.py)
  tl   answer = the text after the last "Answer:" of <vid>_<qid>_tl, up to "Time:" (joint timeline protocol)
  gt   calibration: answer = the ground-truth answer text (expect ~100% yes)
  mcq  calibration: answer = the option text chosen in a multiple-choice run (agreement with letter correctness)
An empty answer is "no" without a judge call. Verdicts are cached by prompt hash (judge_open_cache.jsonl).
Usage (judge venv): python -u judge_open.py --jobs jobs.txt [--tp 1|2]   (1 GPU: the FP8 27B model fits a 40 GB A100 with
short sequences; same weights, greedy decoding, so verdicts match the 2-GPU setting up to numerics)
"""
import argparse
import csv
import hashlib
import json
import os
import re

D = "/nfs-stor/chieu.nguyen/nextgqa"
REPO = "/home/chieu.nguyen/hermes_pool/nextgqa/repo/datasets/nextgqa"
CACHE = "/home/chieu.nguyen/hermes_pool/nextgqa/judge_open_cache.jsonl"
MODEL = "Qwen/Qwen3.8-27B-FP8"
SYSTEM = ("You are an intelligent chatbot designed for evaluating the correctness of generative outputs for "
          "question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if "
          "they match meaningfully. Here's how you can accomplish the task:"
          "------"
          "##INSTRUCTIONS: "
          "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
          "- Consider synonyms or paraphrases as valid matches.\n"
          "- Evaluate the correctness of the prediction compared to the answer.")
USER = ("Please evaluate the following video-based question-answer pair:\n\n"
        "Question: {question}\n"
        "Correct Answer: {answer}\n"
        "Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 "
        "indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where "
        "value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}.")


def cap(s):
    return s[0].upper() + s[1:] if s else s


def meta():
    """base id -> (question, ground-truth answer, options) for the 1,005 test questions."""
    rows = {(r["video_id"], r["qid"]): r for r in csv.DictReader(open(f"{REPO}/test.csv"))}
    out = {}
    for line in open(f"{D}/nextgqa_test_oe.jsonl"):
        s = json.loads(line)
        if s["question_id"].endswith("_oe"):
            m = s["nextgqa"]
            r = rows[(m["vid"], m["qid"])]
            out[s["question_id"][:-3]] = (m["question_text"], m["answer_text"], [cap(r[f"a{i}"]) for i in range(5)])
    return out


def answer_line(reply):
    """The text after the last 'Answer:' up to 'Time:' or the end of that line."""
    if "Answer:" not in reply:
        return ""
    text = reply.rsplit("Answer:", 1)[1]
    text = text.split("Time:", 1)[0]
    return text.strip().split("\n")[0].strip()


def check_ans(response):
    """VideoMind's letter check (as score.py)."""
    b = response.lower().split(" ")[0].replace("(", "").replace(")", "").replace(".", "") if response.strip() else ""
    b = b[:1]
    return b.upper() if b in "abcde" and b else None


def predictions(mode, run_dir, info):
    preds = {}
    if run_dir != "-":
        for r in csv.DictReader(open(f"{run_dir}/results.csv")):
            preds[r["question_id"]] = r["pred_raw"]
    out = {}
    for base, (q, gt, options) in info.items():
        if mode == "oe":
            out[base] = preds.get(f"{base}_oe", "").strip()
        elif mode == "tl":
            out[base] = answer_line(preds.get(f"{base}_tl", ""))
        elif mode == "gt":
            out[base] = gt
        elif mode == "mcq":
            letter = check_ans(preds.get(f"{base}_ans", ""))
            out[base] = options[ord(letter) - ord("A")] if letter else ""
    return out


def parse(text):
    m = re.search(r"['\"]pred['\"]\s*:\s*['\"](yes|no)['\"]", text, re.I)
    s = re.search(r"['\"]score['\"]\s*:\s*([0-9.]+)", text)
    return (m.group(1).lower() if m else "unparsed"), (float(s.group(1)) if s else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()
    info = meta()
    jobs = [l.split() for l in open(args.jobs) if l.strip() and not l.startswith("#")]
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            e = json.loads(line)
            cache[e["key"]] = e
    plan, todo = [], {}
    for mode, run_dir, out_path in jobs:
        rows = []
        for base, pred in predictions(mode, run_dir, info).items():
            q, gt, _ = info[base]
            prompt = USER.format(question=q, answer=gt, pred=pred)
            key = hashlib.sha1((SYSTEM + prompt).encode()).hexdigest()
            rows.append((base, pred, key))
            if pred and key not in cache:
                todo[key] = prompt
        plan.append((mode, run_dir, out_path, rows))
    print(f"{len(jobs)} jobs, {len(todo)} prompts to judge ({len(cache)} cached)", flush=True)
    if todo:
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams
        llm = LLM(model=snapshot_download(MODEL, local_files_only=True), tensor_parallel_size=args.tp,
                  max_model_len=2048 if args.tp > 1 else 1024, max_num_seqs=128 if args.tp > 1 else 32,
                  gpu_memory_utilization=0.88 if args.tp > 1 else 0.94, enforce_eager=True, seed=2024,
                  enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        tok = llm.get_tokenizer()
        keys = sorted(todo)
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": todo[k]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for k in keys]
        outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=40), use_tqdm=False)
        with open(CACHE, "a") as f:
            for k, o in zip(keys, outs):
                verdict, score = parse(o.outputs[0].text)
                cache[k] = {"key": k, "raw": o.outputs[0].text, "verdict": verdict, "score": score}
                f.write(json.dumps(cache[k]) + "\n")
    for mode, run_dir, out_path, rows in plan:
        with open(out_path, "w") as f:
            yes = 0
            for base, pred, key in rows:
                e = cache.get(key) if pred else None
                verdict = e["verdict"] if e else "no"
                yes += verdict == "yes"
                f.write(json.dumps({"key": base, "pred": pred, "verdict": verdict,
                                    "score": e["score"] if e else 0}) + "\n")
        print(f"{mode} {run_dir}: {yes}/{len(rows)} yes -> {out_path}", flush=True)
    print("OPEN JUDGE DONE", flush=True)


if __name__ == "__main__":
    main()
