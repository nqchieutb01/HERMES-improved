#!/bin/bash
# After the phase 43 runs (gold-time upper bounds) and the TimeLens-8B S-EMBER runs finish, judge their S-EMBER runs and the 10% references in one fresh session
# (answer_judgments_fresh43.jsonl). Retries the judge job until it prints FRESH DONE.
cd /home/chieu.nguyen/HERMES
S=/home/chieu.nguyen/hermes_pool/specs
true
echo "runs done $(date +%H:%M)"
for attempt in 1 2 3; do
  sbatch --wait -p cscc-gpu-p --qos=cscc-gpu-qos --gres=gpu:2 --cpus-per-task=16 --mem=64G --time=06:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt) --job-name=h-judge-fresh43 --output=$S/judge_p43.out \
    --export=ALL,RUNS=$S/judge_p43_runs.txt,OUT=answer_judgments_fresh43.jsonl \
    /home/chieu.nguyen/hermes_pool/judge_job.sh
  grep -q "FRESH DONE" $S/judge_p43.out && { echo "JUDGE DONE $(date +%H:%M)"; exit 0; }
  echo "judge attempt $attempt failed $(date +%H:%M)"; sleep 120
done
echo "JUDGE FAILED"
