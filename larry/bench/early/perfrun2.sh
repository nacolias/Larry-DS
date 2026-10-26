#!/bin/bash
# perfrun2.sh NAME [engine args] - perf with DWARF call graphs on the engine's MAIN thread for 2 s of decode
N=$1; shift
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
$XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null &
PID=$!
for i in $(seq 1 300); do grep -q "^prompt:" logs/$N.err 2>/dev/null && break; sleep 1; done
sleep 1
perf record -F 999 --call-graph dwarf,32768 -t $PID -o /tmp/$N.perf.data -- sleep 2 2> /dev/null
wait $PID
echo "rc=$?" >> logs/$N.err
perf report -i /tmp/$N.perf.data --children --percentage relative --sort symbol --stdio -g none 2>/dev/null | grep -v "^#" | grep "%" | head -80 > logs/$N.children.txt
