#!/bin/bash
# Run the 4 embedding shards, retrying while the QoS job-submit limit is reached.
cd /home/chieu.nguyen/HERMES
for s in 0 1 2 3; do
  (
    for attempt in $(seq 1 120); do
      srun --partition=long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=01:30:00 \
        --exclude=$(paste -sd, logs/excluded_nodes.txt),gpu-24,gpu-62 --job-name=h-embed-c$s \
        bash logs/phase10/embed_job.sh --shard $s --num_shards 4 >| logs/phase10/embed-$s.log 2>&1
      grep -q "^DONE" logs/phase10/embed-$s.log && break
      sleep 60
    done
  ) &
  sleep 5
done
wait
echo "ALL EMBED DONE"
