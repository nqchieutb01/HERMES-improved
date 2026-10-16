#!/bin/bash
# Re-extract QVHighlights shard 02 (its first extraction hit NFS "Stale file handle" errors) once all downloads end.
D=/nfs-stor/chieu.nguyen/timelens-bench
until grep -q "ALL DOWNLOADS DONE" /home/chieu.nguyen/hermes_pool/timelens/download_all.log; do sleep 60; done
for s in 02 03; do
  f=$D/qvhighlights_shard_$s.tar.gz
  [ -f $f ] || continue
  tar xzf $f -C $D/videos/qvhighlights && touch $D/videos/qvhighlights/.qvhighlights_shard_$s.tar.gz.extracted && echo "shard $s extracted"
done
echo "qvhighlights videos: $(find $D/videos/qvhighlights -name '*.mp4' | wc -l) (expected 1511)"
