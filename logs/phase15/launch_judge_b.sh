#!/bin/bash
# Waits for phase 15 to finish, then runs the official-protocol answer judge on its runs (2 GPUs).
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" logs/phase15/main_b.log; do sleep 60; done
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh --runs logs/phase15/judge_runs_b.txt >| logs/phase15/judge_b.log 2>&1
  grep -q "^DONE" logs/phase15/judge_b.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase15/judge_b.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase15/judge_b.log; }
  sleep 60
done
echo "JUDGE FINISHED"
