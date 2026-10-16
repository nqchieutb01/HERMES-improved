#!/bin/bash
# TimeLens-8B weights first (needed for the subset test), then the rest of TimeLens-Bench (QVH shards 2-3, Charades,
# ActivityNet 1-6), each file resumable and extracted after it completes. Logs "<item> DONE" lines.
set -u
M=/nfs-stor/chieu.nguyen/models/TimeLens-8B
D=/nfs-stor/chieu.nguyen/timelens-bench
HF=https://huggingface.co
get() {  # get <url> <dest> <expected size>
  until [ "$(stat -c %s "$2" 2>/dev/null)" = "$3" ]; do curl -sSL -C - -o "$2" "$1" || sleep 30; done
}
mkdir -p $M
T=$(cat /scratch/chieu.nguyen/cache/huggingface/token)
curl -s -H "Authorization: Bearer $T" "$HF/api/models/TencentARC/TimeLens-8B/tree/main" | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['type'] == 'file': print(x['path'], x['size'])" | while read f s; do
  get "$HF/TencentARC/TimeLens-8B/resolve/main/$f" "$M/$f" "$s"
done
echo "MODEL DONE $(date +%H:%M)"
curl -s -H "Authorization: Bearer $T" "$HF/api/datasets/TencentARC/TimeLens-Bench/tree/main?recursive=true" | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['type'] == 'file' and x['path'].endswith('.tar.gz'): print(x['path'], x['size'])" | while read f s; do
  name=$(basename $f); ds=$(basename $(dirname $f))
  [ -f "$D/videos/$ds/.$name.extracted" ] && continue
  get "$HF/datasets/TencentARC/TimeLens-Bench/resolve/main/$f" "$D/$name" "$s"
  mkdir -p $D/videos/$ds && tar xzf "$D/$name" -C $D/videos/$ds && touch "$D/videos/$ds/.$name.extracted" && rm -f "$D/$name"
  echo "$name DONE $(date +%H:%M)"
done
echo "ALL DOWNLOADS DONE $(date +%H:%M)"
