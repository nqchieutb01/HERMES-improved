#!/bin/bash
# Report failures of phase 40 TimeLens-script tasks and requeue them via enqueue_full_tl.sh; 3 failures = given up.
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
declare -A tries
while true; do
  requeue=0
  for f in $Q/failed/*-p40-tl-*.json; do
    [ -f "$f" ] || continue
    name=$(basename $f); tries[$name]=$(( ${tries[$name]:-0} + 1 ))
    echo "$(date +%H:%M) failed $name (failure ${tries[$name]}): $(python3 -c "import json; print(json.load(open('$f')).get('last_line', '')[:200])")"
    rm -f "$f"; [ ${tries[$name]} -ge 3 ] && touch $T/runs_full/giveup-$name && echo "giving up on $name"
    requeue=1
  done
  # Claims whose Slurm job is gone (time limit, cancelled, node lost) without a failure report: back to the queue.
  alive=$(squeue --me -h -o %i 2>/dev/null) && for f in $Q/claimed/*-p40-tl-*.json; do
    [ -f "$f" ] || continue
    job=$(python3 -c "import json; print(json.load(open('$f')).get('job', ''))" 2>/dev/null)
    if [ -n "$job" ] && ! grep -qx "$job" <<< "$alive"; then
      echo "$(date +%H:%M) orphaned claim $(basename $f) (job $job gone): requeued"; rm -f "$f"; requeue=1
    fi
  done
  [ $requeue = 1 ] && bash $T/enqueue_full_tl.sh
  sleep 60
done
