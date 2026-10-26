#!/bin/bash
# utrace.sh NAME SECS [engine args] - one-shot engine run under unitrace: device timeline of every card (chrome json in
# logs/ut-NAME/), traced for SECS seconds once decode starts.  Serialised on the one-shot lock like run.sh.
N=$1; S=$2; shift 2
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
UT=/opt/src/pti-gpu/tools/unitrace/build/unitrace
rm -rf logs/ut-$N; mkdir -p logs/ut-$N
exec 9> /run/lock/ds41-oneshot.lock; flock -w 7200 9
timeout ${RUN_TIMEOUT:-1500} systemd-run --scope -q -p MemoryMax=112G -p MemorySwapMax=0 \
  $UT --session $N --start-paused --chrome-kernel-logging --chrome-no-thread-on-device --output-dir-path logs/ut-$N \
  $XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null &
PID=$!
for i in $(seq 1 400); do grep -q "^prompt:" logs/$N.err 2>/dev/null && break; sleep 1; done
sleep 0.5; $UT --resume $N; sleep $S; $UT --pause $N
wait $PID
echo "rc=$?" >> logs/$N.err
