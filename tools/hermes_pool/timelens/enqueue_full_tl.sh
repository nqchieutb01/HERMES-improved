#!/bin/bash
# Phase 40 TimeLens-script tasks on the full benchmark (idempotent; skips queued, running, done, given-up tasks).
# Lines of tl_full_tasks.txt: <priority> <model tag> <model path> <dataset> <run tag> <fps> <total> <min> <chunks>
T=/home/chieu.nguyen/hermes_pool/timelens
Q=/home/chieu.nguyen/hermes_pool/state
while read prio mt model ds tag fps total min chunks; do
  [ -z "$prio" ] && continue
  for ((i = 0; i < chunks; i++)); do
    name=$prio-p40-tl-$mt-$ds-$tag-c$i.json
    done_file=$T/runs_full/$mt/$ds/$tag/$ds-timelens_$i.jsonl
    # Any priority prefix counts (priorities get re-ranked while tasks wait).
    ls $Q/queue/*-p40-tl-$mt-$ds-$tag-c$i.json $Q/claimed/*-p40-tl-$mt-$ds-$tag-c$i.json >/dev/null 2>&1 && continue
    [ -s $done_file ] || ls $T/runs_full/giveup-*-p40-tl-$mt-$ds-$tag-c$i.json >/dev/null 2>&1 && continue
    python3 -c "import json,sys; json.dump({'name': sys.argv[1], 'code': '$T', 'env': {}, 'cmd': sys.argv[2], 'log': sys.argv[3],
    'csv': sys.argv[4], 'est_min': 40.0, 'attempt': 1}, open(sys.argv[5], 'w'))" "${name%.json}" \
      "bash $T/anchor_full.sh $mt $model $ds $tag $fps $total $min $chunks $i" "$T/runs_full/logs/${name%.json}.log" \
      "$done_file" "$Q/queue/$name"
  done
done < $T/tl_full_tasks.txt
