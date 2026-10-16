#!/bin/bash
# (a) batch size 1 through the batched decoder vs the original decoder; (b) batch size 8 with per-step score margins,
# to see whether answers that differ from batch size 1 diverge at near-ties. Reference outputs: logs/batching/out.
cd /home/chieu.nguyen/HERMES
[ -x /l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3 ] || { echo "no /l on $(hostname)"; exit 3; }
echo host=$(hostname); nvidia-smi --query-gpu=name --format=csv,noheader | head -1
PY=/l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3
COMMON=$(grep '^COMMON=' logs/batching/test_job.sh | sed 's/^COMMON=//; s/^"//; s/"$//' | sed 's/\$NV/12/')
PCD="--contrastive_mode blind --contrastive_alpha 1.0 --contrastive_adaptive true --contrastive_scope time"
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
OUT=/home/chieu.nguyen/HERMES/logs/batching/out2
rm -rf $OUT; mkdir -p $OUT
for name in base pcd; do
  extra=""; [ $name = pcd ] && extra="$PCD"
  $PY logs/batching/stage_timer.py $COMMON $extra --batch_size 1 --force_batched_decoder true --decode_log true --save_dir $OUT/$name-forced1 > $OUT/$name-forced1.log 2>&1
  $PY logs/batching/stage_timer.py $COMMON $extra --batch_size 8 --decode_log true --save_dir $OUT/$name-bs8 > $OUT/$name-bs8.log 2>&1
  $PY logs/batching/stage_timer.py $COMMON $extra --batch_size 8 --prefetch_videos 4 --save_dir $OUT/$name-bs8-prefetch > $OUT/$name-bs8-prefetch.log 2>&1
  grep -h -A12 Traceback $OUT/$name-*.log | head -20
  echo "## $name: original bs1 vs batched decoder bs1"; $PY logs/batching/compare.py logs/batching/out/$name-bs1 $OUT/$name-forced1 | head -1
  echo "## $name: original bs1 vs bs8 (rerun)"; $PY logs/batching/compare.py logs/batching/out/$name-bs1 $OUT/$name-bs8 | head -1
  echo "## $name: bs8 run 1 vs bs8 run 2"; $PY logs/batching/compare.py logs/batching/out/$name-bs8 $OUT/$name-bs8 | head -1
  echo "## $name: bs8 vs bs8 + prefetch (expected identical)"; $PY logs/batching/compare.py $OUT/$name-bs8 $OUT/$name-bs8-prefetch | head -1
  for r in bs8 bs8-prefetch; do echo "## $name $r profile"; grep -A10 "=== PROFILE" $OUT/$name-$r.log | head -4; done
  $PY logs/batching/divergence.py logs/batching/out/$name-bs1 $OUT/$name-bs8
  $PY logs/batching/divergence.py logs/batching/out/$name-bs1 $OUT/$name-forced1
done
