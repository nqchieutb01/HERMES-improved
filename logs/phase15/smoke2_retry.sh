#!/bin/bash
cd /home/chieu.nguyen/HERMES
for a in $(seq 1 120); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=01:00:00 --job-name=h-smoke15b logs/phase15/smoke2.sh >| logs/phase15/smoke2.log 2>&1
  grep -q "case 14" logs/phase15/smoke2.log && break
  sleep 60
done
echo SMOKE2 DONE
