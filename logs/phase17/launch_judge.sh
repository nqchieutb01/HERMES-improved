#!/bin/bash
# Waits for phase 17 to finish, then runs the official-protocol answer judge on its runs (2 GPUs).
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" logs/phase17/main.log; do sleep 60; done
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh --runs logs/phase17/judge_runs.txt >| logs/phase17/judge.log 2>&1
  grep -q "^DONE" logs/phase17/judge.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase17/judge.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase17/judge.log; }
  sleep 60
done
echo "JUDGE FINISHED"
