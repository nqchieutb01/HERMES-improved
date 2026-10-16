#!/bin/bash
# Phase 33 chain: wait for the QVHighlights download, extract, build the 500-query subset, build the SigLIP 2 index
# (1-GPU batch job, retried until every subset video and the question file are indexed), then run the phase 33 manager.
cd /home/chieu.nguyen/HERMES
D=/nfs-stor/chieu.nguyen/qvhighlights
S=/home/chieu.nguyen/hermes_pool/specs
PY=/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3
until grep -q "DOWNLOAD DONE" $S/qvh_download.log 2>/dev/null; do sleep 60; done
if [ ! -s $D/qvh_val_single_500.jsonl ]; then
  if [ ! -f $D/videos/.extracted ]; then
    mkdir -p $D/videos
    tar xzf $D/videos.tar.gz -C $D/videos || { echo "extract failed"; exit 1; }
    touch $D/videos/.extracted
  fi
  echo "videos: $(find $D/videos -name '*.mp4' | wc -l) $(date +%H:%M)"
  $PY $S/prepare_qvh.py || { echo "prepare failed"; exit 1; }
fi
need=$(python3 -c "import json; print(len({json.loads(l)['video_id'] for l in open('$D/qvh_val_single_500.jsonl')}))")
have() { ls $D/siglip2_1fps 2>/dev/null | grep -c "\.npz$"; }
# The index is a 1-GPU task in the pool queue (priority 000), so it takes the next free pool GPU.
TASK=000-p33-embed-qvh.json
Q=/home/chieu.nguyen/hermes_pool/state
until [ "$(have)" -ge $((need + 1)) ] && [ -s $D/siglip2_1fps/questions_grounding.npz ] && [ -f $D/siglip2_1fps/.done ]; do
  rm -f $Q/failed/$TASK
  if [ ! -f $Q/queue/$TASK ] && [ ! -f $Q/claimed/$TASK ]; then
    python3 - <<PYEOF
import json
json.dump({"name": "p33-embed-qvh", "code": "/home/chieu.nguyen/HERMES", "est_min": 30, "attempt": 1,
           "env": {"HF_HOME": "/nfs-stor/chieu.nguyen/.cache/huggingface", "HF_HUB_OFFLINE": "1"},
           "cmd": "$PY -u $S/embed_qvh.py && touch $D/siglip2_1fps/.done && echo ok > $S/qvh_embed.done",
           "log": "$S/qvh_embed.out", "csv": "$S/qvh_embed.done"}, open("$Q/queue/$TASK", "w"))
PYEOF
    echo "queued index task $(date +%H:%M)"
  fi
  sleep 60
done
echo "index: $(have) of $((need + 1)) files $(date +%H:%M)"
python3 $S/p33_qvh.py > $S/p33_qvh.json
python3 /home/chieu.nguyen/hermes_pool/pool.py run $S/p33_qvh.json > $S/p33_qvh.log 2>&1
echo "PHASE 33 DONE"
