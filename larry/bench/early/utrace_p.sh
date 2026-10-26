#!/bin/bash
# utrace_p.sh NAME DELAY SECS [engine args] - one-shot run under unitrace, traced for SECS seconds starting DELAY
# seconds after the prompt's first batch begins (the prompt path's kernels on every card).  One engine per box.
N=$1; DL=$2; S=$3; shift 3
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
if systemctl is-active --quiet larry-ds41; then echo "utrace_p.sh: larry-ds41 is running - stop it first" >&2; exit 3; fi
UT=/opt/src/pti-gpu/tools/unitrace/build/unitrace
rm -rf logs/ut-$N; mkdir -p logs/ut-$N
exec 9> /run/lock/ds41-oneshot.lock; flock -w 7200 9
timeout ${RUN_TIMEOUT:-1500} systemd-run --scope -q -p MemoryMax=112G -p MemorySwapMax=0 \
  $UT --session $N --start-paused --chrome-kernel-logging --chrome-no-thread-on-device --output-dir-path logs/ut-$N \
  $XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null &
PID=$!
for i in $(seq 1 600); do grep -q "enc-skip: first batch" logs/$N.err 2>/dev/null && break; sleep 0.5; done
sleep $DL; $UT --resume $N; sleep $S; $UT --pause $N
wait $PID
echo "rc=$?" >> logs/$N.err
