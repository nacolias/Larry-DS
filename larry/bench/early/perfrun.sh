#!/bin/bash
# perfrun.sh NAME [engine args] - one-shot engine run; perf samples the main thread during decode (started 75 s in,
# after placement + prompt, for 5 s)
N=$1; shift
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
$XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null &
PID=$!
# wait for the prompt to be done (decode starts): poll the err log for "prompt:"
for i in $(seq 1 300); do grep -q "^prompt:" logs/$N.err 2>/dev/null && break; sleep 1; done
sleep 1
perf record -F 2000 -g -p $PID -o /tmp/$N.perf.data -- sleep 4 2> /dev/null
wait $PID
echo "rc=$?" >> logs/$N.err
perf report -i /tmp/$N.perf.data --no-children --sort dso,symbol --stdio 2>/dev/null | grep -v "^#" | grep "%" | head -45 > logs/$N.perf.txt
perf report -i /tmp/$N.perf.data --no-children --sort comm,tid --stdio 2>/dev/null | grep "%" | head -20 > logs/$N.perf.threads.txt
