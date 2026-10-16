#!/bin/bash
# Phase 37 launcher: TimeLens-script runs (anchor + simple baselines) and the SigLIP index as pool tasks, our
# no-retrieval runs (p37a) now, our retrieval runs (p37b) once the index is complete.
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
D=/nfs-stor/chieu.nguyen/timelens-bench
PY=/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3
enqueue() {  # enqueue <task file> <name> <cmd> <log> <done file> <est_min>
  [ -f $Q/queue/$1 ] || [ -f $Q/claimed/$1 ] || [ -s "$5" ] && return
  python3 -c "import json,sys; json.dump({'name': sys.argv[1], 'code': '$T', 'env': {}, 'cmd': sys.argv[2], 'log': sys.argv[3],
  'csv': sys.argv[4], 'est_min': float(sys.argv[5]), 'attempt': 1}, open(sys.argv[6], 'w'))" "$2" "$3" "$4" "$5" "$6" "$Q/queue/$1"
}
mkdir -p $T/runs
# SigLIP 2 index for retrieval (priority 000).
enqueue 000-p37-embed.json p37-embed \
  "HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface HF_HUB_OFFLINE=1 $PY -u $T/embed_tl.py && touch $D/siglip2_1fps/.done && echo ok > $T/runs/embed.done" \
  $T/runs/embed.out $T/runs/embed.done 30
# TimeLens' own evaluation: tag fps total_tokens min_tokens (8 chunks each, priority 020).
for cfg in "anchor 2 14336 64" "tl-2fps-4500 2 4500 16" "tl-1fps-5000 1 5000 16" "tl-1fps-1365 1 1365 16"; do
  set -- $cfg
  for i in 0 1 2 3 4 5 6 7; do
    enqueue 020-p37-tl-$1-c$i.json p37-tl-$1-c$i "bash $T/anchor_chunk.sh $1 $2 $3 $4 8 $i" \
      $T/runs/$1-c$i.log $T/runs/$1/qvhighlights-timelens_$i.jsonl 30
  done
done
echo "queued TimeLens-script tasks $(date +%H:%M)"
python3 /home/chieu.nguyen/hermes_pool/pool.py run $T/p37a.json >> $T/p37a.log 2>&1 &
until [ -f $D/siglip2_1fps/.done ]; do sleep 60; done
echo "index done $(date +%H:%M)"
python3 /home/chieu.nguyen/hermes_pool/pool.py run $T/p37b.json >> $T/p37b.log 2>&1
wait
echo "PHASE 37 DONE"
