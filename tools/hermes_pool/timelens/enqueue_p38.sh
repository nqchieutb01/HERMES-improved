#!/bin/bash
# Phase 38 TimeLens-script tasks for TimeLens-8B (idempotent; skips queued, running, done and given-up tasks).
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
M=/nfs-stor/chieu.nguyen/models/TimeLens-8B
enqueue() {  # enqueue <task file> <name> <cmd> <log> <done file> <est_min>
  [ -f $Q/queue/$1 ] || [ -f $Q/claimed/$1 ] || [ -s "$5" ] || [ -f $T/runs/giveup-$1 ] && return
  python3 -c "import json,sys; json.dump({'name': sys.argv[1], 'code': '$T', 'env': {}, 'cmd': sys.argv[2], 'log': sys.argv[3],
  'csv': sys.argv[4], 'est_min': float(sys.argv[5]), 'attempt': 1}, open(sys.argv[6], 'w'))" "$2" "$3" "$4" "$5" "$6" "$Q/queue/$1"
}
for cfg in "t8-anchor 2 14336 64" "t8-tl-2fps-4500 2 4500 16" "t8-tl-1fps-1365 1 1365 16"; do
  set -- $cfg
  for i in 0 1 2 3 4 5 6 7; do
    enqueue 020-p38-tl-$1-c$i.json p38-tl-$1-c$i "bash $T/anchor_chunk.sh $1 $2 $3 $4 8 $i $M" \
      $T/runs/$1-c$i.log $T/runs/$1/qvhighlights-timelens_$i.jsonl 30
  done
done
