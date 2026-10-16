#!/bin/bash
cd /home/chieu.nguyen/HERMES
for attempt in $(seq 1 200); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:2 --cpus-per-task=16 --mem=80G --time=03:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-judge \
    bash logs/phase11/judge_job.sh >| logs/phase11/judge.log 2>&1
  grep -q "^DONE" logs/phase11/judge.log && break
  grep -q "SubmitJob\|Unable to allocate" logs/phase11/judge.log || { echo "judge attempt $attempt failed"; tail -5 logs/phase11/judge.log; }
  sleep 60
done
echo "JUDGE FINISHED"
