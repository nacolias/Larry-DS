#!/bin/bash
# tempab.sh [PFX] - the sampling temperature's effect on decode (DSpark / lookup acceptance) and on loops: agentd.py's deep
# agent session (thinking on, top_p 0.95) at temperature 1.0 / 0.6 / 1.0 / 0.6 on private servers with the service's
# config (deeprun.sh), then the loop guards' firings in each run's server log.  An abw.sh probe ("@bash bench/loop/tempab.sh").
cd /opt/ai/dsv41
P=${1:-tmp}
for r in t10a t06a t10b t06b; do
  T=1.0
  [ "${r:1:2}" = 06 ] && T=0.6
  AGENTD_ARGS="--first 64000 --turns 24 --temp $T" bash bench/loop/deeprun.sh $P-$r 2>&1 | grep -E "turns 1-|\] done|STILL|exited|rror"
  echo "[$P-$r] T=$T loop guards: $(grep -c "repeated the same passages" logs/serve-$P-$r.log 2>/dev/null), finishes: $(grep -o "(stop\|(length\|(tool_calls" logs/serve-$P-$r.log 2>/dev/null | sort | uniq -c | tr "\n" " ")"
  [ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
done
