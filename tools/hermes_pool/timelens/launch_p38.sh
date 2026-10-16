#!/bin/bash
# Phase 38: part a now; part b (TimeLens-8B) once its weights are downloaded. TimeLens-script failures are reported
# and requeued (3 failures = given up), as in phase 37.
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
python3 /home/chieu.nguyen/hermes_pool/pool.py run $T/p38a.json >> $T/p38a.log 2>&1 &
until grep -q "MODEL DONE" $T/download_all.log; do sleep 60; done
echo "model ready $(date +%H:%M)"
bash $T/enqueue_p38.sh
python3 /home/chieu.nguyen/hermes_pool/pool.py run $T/p38b.json >> $T/p38b.log 2>&1 &
declare -A tries
until [ "$(ls $T/runs/t8-*/qvhighlights-timelens_*.jsonl 2>/dev/null | wc -l)" -ge 24 ] && ! pgrep -f "pool.py run $T/p38" > /dev/null; do
  requeue=0
  for f in $Q/failed/020-p38-tl-*.json; do
    [ -f "$f" ] || continue
    name=$(basename $f); tries[$name]=$(( ${tries[$name]:-0} + 1 ))
    echo "$(date +%H:%M) failed $name (failure ${tries[$name]}): $(python3 -c "import json; print(json.load(open('$f')).get('last_line', '')[:200])")"
    rm -f "$f"; [ ${tries[$name]} -ge 3 ] && touch $T/runs/giveup-$name && echo "giving up on $name"
    requeue=1
  done
  [ $requeue = 1 ] && bash $T/enqueue_p38.sh
  sleep 60
done
echo "PHASE 38 DONE"
