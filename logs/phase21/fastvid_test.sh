#!/bin/bash
# FastVID smoke test: retention 1.0 through the FastVID path must equal plain unpruned encoding; retention 0.1 must
# keep ~10% of the visual tokens.
cd /home/chieu.nguyen/HERMES
[ -x /l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3 ] || { echo "no /l on $(hostname)"; exit 3; }
echo host=$(hostname)
PY=/l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3
COMMON=$(grep '^COMMON=' logs/batching/test_job.sh | sed 's/^COMMON=//; s/^"//; s/"$//' | sed 's/\$NV/4/' | sed 's/--offline_keep_ratio 0.1 --prune_score random //')
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
OUT=/home/chieu.nguyen/HERMES/logs/phase21/fastvid_test; rm -rf $OUT; mkdir -p $OUT
$PY -m video_qa.hermes_vqa $COMMON --offline_keep_ratio 1.0 --prune_score random --save_dir $OUT/plain > $OUT/plain.log 2>&1
$PY -m video_qa.hermes_vqa $COMMON --offline_keep_ratio 1.0 --prune_score fastvid --save_dir $OUT/fv100 > $OUT/fv100.log 2>&1
$PY -m video_qa.hermes_vqa $COMMON --offline_keep_ratio 0.1 --prune_score fastvid --retention_snapshot true --save_dir $OUT/fv10 > $OUT/fv10.log 2>&1
$PY -m video_qa.hermes_vqa $COMMON --offline_keep_ratio 0.1 --prune_score hermes --save_dir $OUT/h10 > $OUT/h10.log 2>&1
grep -h -A12 Traceback $OUT/*.log | head -30
echo "## plain vs fastvid r=1.0 (must be identical)"; $PY logs/batching/compare.py $OUT/plain $OUT/fv100
grep -h "FastVID: kept" $OUT/fv10.log | head -5; grep -h "Offline pruning: keeping" $OUT/h10.log | head -5
grep -h "Answering Cache lengths" $OUT/fv10.log | head -3; grep -h "Answering Cache lengths" $OUT/h10.log | head -3
grep -h "Pred Answer" $OUT/fv10.log | head -4 | cut -c1-200
echo FASTVID TEST DONE
