#!/bin/bash
# ra-ab.sh [BIN] - what patch-roomahead.py's read-ahead costs a 131K prompt: the staged engine (BIN, default rabin) twice,
# read-ahead off (STRATA_TP_AHEAD_S=0: the cards still evict side by side) then on, each with two fresh 131K needles
# (conversation slots 1 and 0, so both grow their replicas), STRATA_BATCH_TRACE=1 for the batches' times, and
# bench/speed/vmwatch.sh's reclaim counters meanwhile (logs/vmwatch-raab.txt).  Engine logs: logs/engine-raab-{off,on}.log.
# An abw.sh "@" step.
cd /opt/ai/dsv41
BIN=${1:-/opt/ai/dsv41/rabin}
bash bench/speed/vmwatch.sh logs/vmwatch-raab.txt 900 &
VW=$!
for V in off on; do
  E="STRATA_BATCH_TRACE=1"
  [ $V = off ] && E="$E STRATA_TP_AHEAD_S=0"
  echo "=== read-ahead $V  $(date +%T)"
  RA_TOKENS=131072,131072 RA_ENV="$E" bash bench/longctx/r1m-ra.sh $BIN 2>&1 | grep -E "needle|deep|turn|faults|read-ahead|gave|KV replicas" | cut -c1-230
  cp logs/engine-r1mA.log logs/engine-raab-$V.log
done
kill $VW 2>/dev/null
