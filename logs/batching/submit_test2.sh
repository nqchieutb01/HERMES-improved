#!/bin/bash
cd /home/chieu.nguyen/HERMES
for i in $(seq 1 120); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=01:30:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt) --job-name=h-batchtest2 bash logs/batching/test_job2.sh >| logs/batching/test2.log 2>&1
  grep -q "host=" logs/batching/test2.log && break
  grep -q "no /l on" logs/batching/test2.log && grep -o "no /l on [a-z0-9-]*" logs/batching/test2.log | awk '{print $4}' >> logs/excluded_nodes.txt; sleep 60
done
echo TEST2 FINISHED >> logs/batching/test2.log
