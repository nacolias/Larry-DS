#!/bin/bash
# tperf.sh - host profile of plain decode: perf on the engine's main thread for 8 s mid-decode (flat, then call graph)
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
N=perf1
env $B RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 1200 > /dev/null 2>&1 &
RP=$!
until grep -q "^prompt:" logs/$N.err 2>/dev/null; do sleep 1; done
sleep 3
P=$(pgrep -x ds41_generate_x | head -1)
[ -z "$P" ] && P=$(pgrep -f "[d]s41_generate_xe -m" | head -1)
echo "engine pid $P"
perf record -o logs/perf1.data -t $P -F 4999 -- sleep 6 > /dev/null 2>&1
perf record -o logs/perf1g.data -t $P -F 1999 --call-graph dwarf,16384 -- sleep 3 > /dev/null 2>&1
wait $RP
echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
perf report -i logs/perf1.data --no-children --sort dso -q 2>/dev/null | head -15
echo ---
perf report -i logs/perf1.data --no-children --sort dso,symbol -q 2>/dev/null | head -60
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
