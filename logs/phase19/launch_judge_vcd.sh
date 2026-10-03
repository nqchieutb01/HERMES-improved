#!/bin/bash
# Waits for phase 18 to finish, then runs the official-protocol answer judge on its runs (2 GPUs).
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" logs/phase19/vcd.log; do sleep 60; done
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh --runs logs/phase19/judge_runs_vcd.txt >| logs/phase19/judge_vcd.log 2>&1
  # A node without the judge venv: exclude it before retrying.
  grep -q "judge venv missing on" logs/phase19/judge_vcd.log && grep -o "missing on [a-z0-9-]*" logs/phase19/judge_vcd.log | awk '{print $3}' >> logs/excluded_nodes.txt
  grep -q "^DONE" logs/phase19/judge_vcd.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase19/judge_vcd.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase19/judge_vcd.log; }
  sleep 60
done
echo "JUDGE FINISHED"
