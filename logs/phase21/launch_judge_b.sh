#!/bin/bash
# Waits for phase 18 to finish, then runs the official-protocol answer judge on its runs (2 GPUs).
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" logs/phase21/b.log; do sleep 60; done
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh --runs logs/phase21/judge_runs_b.txt >| logs/phase21/judge_b.log 2>&1
  # A node without the judge venv: exclude it before retrying.
  grep -q "judge venv missing on" logs/phase21/judge_b.log && grep -o "missing on [a-z0-9-]*" logs/phase21/judge_b.log | awk '{print $3}' >> logs/excluded_nodes.txt
  grep -q "^DONE" logs/phase21/judge_b.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase21/judge_b.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase21/judge_b.log; }
  sleep 60
done
echo "JUDGE FINISHED"
