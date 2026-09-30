#!/bin/bash
# Waits for phase 16 to finish, then runs the official-protocol answer judge on its runs (2 GPUs).
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" logs/phase16/uniform.log; do sleep 60; done
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh --runs logs/phase16/judge_runs_uniform.txt >| logs/phase16/judge_uniform.log 2>&1
  grep -q "^DONE" logs/phase16/judge_uniform.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase16/judge_uniform.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase16/judge_uniform.log; }
  sleep 60
done
echo "JUDGE FINISHED"
