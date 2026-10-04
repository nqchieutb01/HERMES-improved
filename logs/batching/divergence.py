"""Where a batched run's answer leaves the reference answer, and how close the decision was there.

For each question whose answer differs, finds the first generated token after which the batched answer is no longer a
prefix of the reference text, and prints the batched run's margin between its best and second-best score at that step
(decode-<chunk>.jsonl). A margin within the rounding of bf16 logits (~0.1 here) means a near-tie, i.e. a
floating-point effect, not a different computation.
"""
import glob
import json
import statistics
import sys

import pandas as pd
from transformers import AutoTokenizer

ref = pd.read_csv(f"{sys.argv[1]}/1_0.csv").set_index("question_id")
log = {}
for f in glob.glob(f"{sys.argv[2]}/decode-*.jsonl"):
    for line in open(f):
        d = json.loads(line)
        log[d["question_id"]] = d
tok = AutoTokenizer.from_pretrained("/l/users/chieu.nguyen/models/Qwen3-VL-8B-Instruct")
all_margins = [m for d in log.values() for m in d["margins"]]
at_div = []
for q, d in log.items():
    text = str(ref.loc[q, "pred_raw"])
    for step in range(1, len(d["tokens"]) + 1):
        prefix = tok.decode(d["tokens"][:step], skip_special_tokens=True)
        if not text.startswith(prefix):
            at_div.append(d["margins"][step - 1])
            break
print(f"{sys.argv[2]}: {len(at_div)} of {len(log)} answers diverge from the reference; margin at the divergence step: "
      f"median {statistics.median(at_div) if at_div else float('nan'):.3f}, max {max(at_div) if at_div else float('nan'):.3f} "
      f"(all steps: median {statistics.median(all_margins):.2f}, share < 0.15: "
      f"{100 * sum(m < 0.15 for m in all_margins) / len(all_margins):.1f}%)")
