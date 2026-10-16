#!/bin/bash
# Phase 34 chain: download the first ActivityNet video tar part (5.4 GB, resumable) and the val_1 annotations,
# extract the complete videos in it, build the 500-query subset, build the SigLIP 2 index as a pool task, then run
# the phase 34 manager.
cd /home/chieu.nguyen/HERMES
D=/nfs-stor/chieu.nguyen/activitynet
S=/home/chieu.nguyen/hermes_pool/specs
PY=/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3
B=https://huggingface.co/datasets/friedrichor/ActivityNet_Captions/resolve/main
mkdir -p $D/videos
[ -s $D/activitynet_captions_val1.json ] || curl -sSL -o $D/activitynet_captions_val1.json $B/activitynet_captions_val1.json
until [ "$(stat -c %s $D/part-000.tar 2>/dev/null)" = "5368709120" ]; do
  curl -sSL -C - -o $D/part-000.tar $B/ActivityNet_Videos.tar.part-000 || sleep 60
done
echo "download done $(date +%H:%M)"
if [ ! -s $D/anet_val1_500.jsonl ]; then
  # The part ends mid-file: tar reports an unexpected EOF after extracting every complete member.
  [ -f $D/videos/.extracted ] || { tar xf $D/part-000.tar -C $D/videos; touch $D/videos/.extracted; }
  echo "videos: $(find $D/videos -name '*.mp4' | wc -l) $(date +%H:%M)"
  $PY $S/prepare_anet.py || { echo "prepare failed"; exit 1; }
fi
need=$(python3 -c "import json; print(len({json.loads(l)['video_id'] for l in open('$D/anet_val1_500.jsonl')}))")
have() { ls $D/siglip2_1fps 2>/dev/null | grep -c "\.npz$"; }
TASK=000-p34-embed-anet.json
Q=/home/chieu.nguyen/hermes_pool/state
until [ "$(have)" -ge $((need + 1)) ] && [ -s $D/siglip2_1fps/questions_grounding.npz ] && [ -f $D/siglip2_1fps/.done ]; do
  rm -f $Q/failed/$TASK
  if [ ! -f $Q/queue/$TASK ] && [ ! -f $Q/claimed/$TASK ]; then
    python3 - <<PYEOF
import json
json.dump({"name": "p34-embed-anet", "code": "/home/chieu.nguyen/HERMES", "est_min": 30, "attempt": 1,
           "env": {"HF_HOME": "/nfs-stor/chieu.nguyen/.cache/huggingface", "HF_HUB_OFFLINE": "1"},
           "cmd": "$PY -u $S/embed_anet.py && touch $D/siglip2_1fps/.done && echo ok > $S/anet_embed.done",
           "log": "$S/anet_embed.out", "csv": "$S/anet_embed.done"}, open("$Q/queue/$TASK", "w"))
PYEOF
    echo "queued index task $(date +%H:%M)"
  fi
  sleep 60
done
echo "index: $(have) of $((need + 1)) files $(date +%H:%M)"
python3 $S/p34_anet.py > $S/p34_anet.json
python3 /home/chieu.nguyen/hermes_pool/pool.py run $S/p34_anet.json > $S/p34_anet.log 2>&1
echo "PHASE 34 DONE"
