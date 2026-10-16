#!/bin/bash
cd /home/chieu.nguyen/HERMES
for i in $(seq 1 120); do
  srun --partition=long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=01:00:00 \
    --exclude=$(paste -sd, logs/excluded_nodes.txt) --job-name=h-fastvidtest bash logs/phase21/fastvid_test.sh >| logs/phase21/fastvid_test.log 2>&1
  grep -q "host=" logs/phase21/fastvid_test.log && break
  grep -q "no /l on" logs/phase21/fastvid_test.log && grep -o "no /l on [a-z0-9-]*" logs/phase21/fastvid_test.log | awk '{print $4}' >> logs/excluded_nodes.txt
  sleep 60
done
echo SUBMIT DONE >> logs/phase21/fastvid_test.log
