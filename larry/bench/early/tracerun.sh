#!/bin/bash
# tracerun.sh NAME [engine args] - one-shot engine run; `perf trace -s` summarises the syscalls of the decode phase (3 s)
N=$1; shift
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
$XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null &
PID=$!
for i in $(seq 1 300); do grep -q "^prompt:" logs/$N.err 2>/dev/null && break; sleep 1; done
sleep 1
timeout 10 perf trace -s -p $PID -o logs/$N.trace.txt -- sleep 3 2>/dev/null
wait $PID
echo "rc=$?" >> logs/$N.err
