#!/bin/bash
# TimeLens-Bench QVHighlights shard 01 (5.84 GB), resumable.
D=/nfs-stor/chieu.nguyen/timelens-bench
B=https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/qvhighlights
cd $D
until [ "$(stat -c %s qvhighlights_shard_01.tar.gz 2>/dev/null)" = "5837131978" ]; do
  curl -sSL -C - -o qvhighlights_shard_01.tar.gz $B/qvhighlights_shard_01.tar.gz || sleep 60
done
mkdir -p videos/qvhighlights && tar xzf qvhighlights_shard_01.tar.gz -C videos/qvhighlights && echo "EXTRACT DONE $(date +%H:%M)"
echo "DOWNLOAD DONE $(date +%H:%M)"
