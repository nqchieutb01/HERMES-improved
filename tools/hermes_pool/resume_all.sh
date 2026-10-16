#!/bin/bash
# Restart whatever of the pool is not running (after a login-node reboot or a killed process). Safe to run any time:
# each piece starts only if it is not already running and has not finished. Usage: bash resume_all.sh
P=/home/chieu.nguyen/hermes_pool
S=$P/specs
cd $P
running() { ps -eo args | grep -F -- "$1" | grep -vq -e grep -e resume_all; }
start() {  # start <log> <command...>
  local log=$1; shift
  setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
  echo "started: $*"
}
running "pool.py keeper" || start $P/keeper.log python3 $P/pool.py keeper $P/slots.json
for spec in p30_budgets p31_fixes p32_ablations p33b_qvh p34b_anet p35_semvid_reserve p36_frames; do
  grep -q "ALL DONE" $S/$spec.log 2>/dev/null && continue
  running "pool.py run $S/$spec.json" || running "pool.py run specs/$spec.json" || running "../pool.py run $spec.json" \
    || start $S/$spec.log python3 $P/pool.py run $S/$spec.json
done
grep -q "DOWNLOAD DONE" $S/qvh_download.log 2>/dev/null || running "qvh_download.sh" \
  || start $S/qvh_download.log bash $S/qvh_download.sh
# qvh_setup.sh resumes from its own markers (extracted videos, subset file, index) and then runs the p33 manager.
grep -q "PHASE 33 DONE" $S/qvh_setup.log 2>/dev/null || running "qvh_setup.sh" || running "p33_qvh.json" \
  || start $S/qvh_setup.log bash $S/qvh_setup.sh
grep -q "PHASE 34 DONE" $S/anet_setup.log 2>/dev/null || running "anet_setup.sh" || running "p34_anet.json" \
  || start $S/anet_setup.log bash $S/anet_setup.sh
# TimeLens-Bench phases (37-38) and the full-benchmark download.
TL=$P/timelens
for spec in p37a p37b p38a p38b; do
  grep -q "ALL DONE" $TL/$spec.log 2>/dev/null && continue
  running "pool.py run $TL/$spec.json" || running "pool.py run timelens/$spec.json" || start $TL/$spec.log python3 $P/pool.py run $TL/$spec.json
done
for spec in $P/nextgqa/p39_check $P/nextgqa/p39_rest $P/nextgqa/p39_t8 $P/nextgqa/p41 $P/nextgqa/p42_check $P/nextgqa/p42_sweep $P/nextgqa/p42_t8 $P/specs/p43 $TL/p40_check $TL/p40_sweep $TL/p40_sember; do
  [ -f $spec.json ] || continue
  grep -q "ALL DONE" $spec.log 2>/dev/null && continue
  running "pool.py run $spec.json" || running "pool.py run $(basename $spec).json" || start $spec.log python3 $P/pool.py run $spec.json
done
running "retry_p40.sh" || start $TL/retry_p40.log bash $TL/retry_p40.sh
grep -q "ALL DOWNLOADS DONE" $TL/download_all.log 2>/dev/null || running "download_all.sh" \
  || start $TL/download_all.log bash $TL/download_all.sh
for chain in judge_main judge_p30_p31 judge_p32 judge_p43; do
  grep -q "JUDGE DONE" $S/$chain.chain.log 2>/dev/null || running "$chain.sh" \
    || start $S/$chain.chain.log bash $S/$chain.sh
done
echo "resume check done $(date +%H:%M)"
