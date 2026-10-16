#!/bin/bash
# QVHighlights val (jwnt4/qvhighlights-val): annotations + videos.tar.gz (17.6 GB), resumable (curl -C -).
D=/nfs-stor/chieu.nguyen/qvhighlights
B=https://huggingface.co/datasets/jwnt4/qvhighlights-val/resolve/main
cd $D
curl -sSL -o highlight_val_release.jsonl $B/highlight_val_release.jsonl
until [ "$(stat -c %s videos.tar.gz 2>/dev/null)" = "17606382889" ]; do
  curl -sSL -C - -o videos.tar.gz $B/videos.tar.gz || sleep 60
done
echo "DOWNLOAD DONE $(date +%H:%M)"
