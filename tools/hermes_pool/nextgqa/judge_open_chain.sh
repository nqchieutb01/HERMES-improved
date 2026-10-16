#!/bin/bash
# Waits for a pool manager's ALL DONE, then runs judge_open.py on a jobs file in a 2-GPU job (retries 3x).
# Usage: bash judge_open_chain.sh <manager log> <jobs file> <tag>
LOG=$1 JOBS=$2 TAG=$3
N=/home/chieu.nguyen/hermes_pool/nextgqa
cd /home/chieu.nguyen/HERMES
until grep -q "ALL DONE" $LOG; do sleep 60; done
echo "runs done $(date +%H:%M)"
for attempt in 1 2 3; do
  sbatch --wait -p cscc-gpu-p --qos=cscc-gpu-qos --gres=gpu:${GPUS:-2} --cpus-per-task=16 --mem=64G --time=02:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt) --job-name=h-judge-open-$TAG --output=$N/judge_open_$TAG.out \
    --export=ALL,JOBS=$JOBS,TP=${GPUS:-2} $N/judge_open_job.sh
  grep -q "OPEN JUDGE DONE" $N/judge_open_$TAG.out && { echo "JUDGE DONE $(date +%H:%M)"; exit 0; }
  echo "judge attempt $attempt failed $(date +%H:%M)"; sleep 120
done
echo "JUDGE FAILED"
