"""Judge S-EMBER grounding answers with the official S-EMBER protocol, using a local judge model.

Follows facebookresearch/S-EMBER tools/judge_grounding.py (copy in logs/phase11/official/sember_official_judge.py): the same
JUDGE_PROMPT, the same gold list (the `answer` field plus every annotator's `answer_text`; a prediction
is CORRECT if semantically equivalent to ANY ONE of them) and the same verdict parsing
(CORRECT / WRONG / UNPARSED from the first line). The only difference is the judge model: the official
script calls Gemini (gemini-3.1-flash, needs an API key and sends data to an external service); here a
local Qwen3.8-27B-FP8 runs through vLLM at temperature 0.

The prediction passed to the judge is the model's answer text (the "Answer:" part, without the "Time:"
interval). Identical (question, golds, prediction) triples across runs are judged once (resumable cache:
logs/phase11/judge_cache_official.jsonl).

Writes <run>/answer_judgments.jsonl (question_id, verdict, correct) for every run listed in --runs.
Run in the judge environment on a GPU node, from the repo root:
  <judge-venv>/bin/python logs/phase11/judge_grounding.py --runs logs/phase11/runs.txt
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs" / "phase11" / "official"))
os.environ.setdefault("HF_HOME", "/nfs-stor/chieu.nguyen/.cache/huggingface")
import sember_official_judge as official  # noqa: E402  (official S-EMBER script: prompt and gold helpers)

MODEL = "Qwen/Qwen3.8-27B-FP8"
RESULTS = ROOT / "results"
CACHE = ROOT / "logs" / "phase11" / "judge_cache_official.jsonl"


def golds(row):
    record = {"answer": row.get("answer"), "answers": json.loads(row.get("answers_json") or "[]")}
    return official._golds_for(record)


def prediction(row):
    return (row.get("pred_answer_parsed") or row.get("pred_answer") or row.get("pred_raw") or "").strip()


def verdict_of(text):
    """Official parsing of the judge's first line."""
    first_line = text.strip().split("\n", 1)[0].strip().upper()
    if "CORRECT" in first_line and "WRONG" not in first_line:
        return "CORRECT"
    return "WRONG" if "WRONG" in first_line else "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="text file: one run path per line, relative to results/")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    runs = [line.strip() for line in open(args.runs) if line.strip()]
    per_run, todo = {}, {}
    for run in runs:
        items = []
        for line in open(RESULTS / run / "sember_grounding_scored.jsonl"):
            r = json.loads(line)
            prompt = official.JUDGE_PROMPT.format(question=r["question"],
                                                  golds=official._format_golds(golds(r)), pred=prediction(r))
            k = hashlib.sha1(prompt.encode()).hexdigest()
            items.append((r["question_id"], k))
            todo[k] = prompt
        per_run[run] = items

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec
    pending = [k for k in todo if k not in cache]
    print(f"{len(runs)} runs, {len(todo)} unique prompts, {len(pending)} to judge", flush=True)

    if pending:
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams
        path = snapshot_download(args.model, local_files_only=True)
        llm = LLM(model=path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=4096,
                  max_num_seqs=128, gpu_memory_utilization=0.88, enforce_eager=True, seed=2024,
                  enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        tok = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0, max_tokens=64)
        with CACHE.open("a") as out:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                prompts = [tok.apply_chat_template([{"role": "user", "content": todo[k]}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) for k in batch]
                for k, res in zip(batch, llm.generate(prompts, sampling, use_tqdm=False)):
                    text = res.outputs[0].text
                    rec = {"key": k, "verdict": verdict_of(text), "raw": text.strip()[:300]}
                    cache[k] = rec
                    out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"judged {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)

    for run, items in per_run.items():
        with open(RESULTS / run / "answer_judgments.jsonl", "w") as out:
            for qid, k in items:
                v = cache[k]["verdict"]
                out.write(json.dumps({"question_id": qid, "verdict": v, "correct": v == "CORRECT"}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
