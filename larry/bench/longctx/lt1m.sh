#!/bin/bash
# lt1m.sh BIN NAME:VAR=v,VAR=v ... - long prompts under each configuration on a staged engine: bench/longctx/r1m-ra.sh (a
# 131K needle into conversation slot 1 - the fresh engine's first big prompt - then a 984K needle into slot 0, each with
# its decode and a 12K turn, the replica check on), its engine log kept as logs/engine-lt1m-NAME.log.  An abw.sh "@" step.
cd /opt/ai/dsv41
BIN=$1; shift
for CFG in "$@"; do
  N=${CFG%%:*}; E=${CFG#*:}; [ "$E" = "$CFG" ] && E=
  echo "=== $N: ${E//,/ }  $(date +%T)"
  RA_ENV="${E//,/ }" bash bench/longctx/r1m-ra.sh $BIN 2>&1 | grep -E "needle|deep|turn|faults" | cut -c1-230
  cp logs/engine-r1mA.log logs/engine-lt1m-$N.log
  grep -a -E "larry li|prompt phases: [0-9]{6}" logs/engine-lt1m-$N.log | cut -c1-200 | tail -6
done
