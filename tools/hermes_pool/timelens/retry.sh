#!/bin/bash
# Report failures of phase 37 tasks without a manager (index, TimeLens-script chunks) and requeue them through the
# idempotent enqueue_tasks.sh; after 3 failures a task is given up (runs/giveup-<task>). Ends at PHASE 37 DONE.
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
declare -A tries
until grep -q "PHASE 37 DONE" $T/launch.log 2>/dev/null; do
  requeue=0
  for f in $Q/failed/020-p37-tl-*.json $Q/failed/000-p37-embed.json; do
    [ -f "$f" ] || continue
    name=$(basename $f)
    tries[$name]=$(( ${tries[$name]:-0} + 1 ))
    echo "$(date +%H:%M) failed $name (failure ${tries[$name]}): $(python3 -c "import json; print(json.load(open('$f')).get('last_line', '')[:200])")"
    rm -f "$f"
    if [ ${tries[$name]} -ge 3 ]; then echo "giving up on $name"; touch $T/runs/giveup-$name; fi
    requeue=1
  done
  [ $requeue = 1 ] && bash $T/enqueue_tasks.sh
  sleep 60
done
