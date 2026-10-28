#!/bin/bash
# r1m-ra.sh [BIN] - patch-roomahead.py on the staged engine (BIN, default /opt/ai/dsv41/rabin): r1m.sh variant A (a 131K
# needle into conversation slot 1, then a 984K needle into slot 0 - RA_TOKENS - each with its 400-token decode and a 12K
# turn) with the replica sync log + check (RA_ENV: more environment); then what the read-ahead and the evictions logged.
# The time before the first reply is the needles' 9-token decode (r1m-stage.sh on the engine before it: 131K 2.22 s,
# 984K 8.09 s - evictions 3 x ~0.63 s / 1.15 + 4.97 + 0.99 s).  An abw.sh "@" step.
cd /opt/ai/dsv41
BIN=${1:-/opt/ai/dsv41/rabin}
R1M_VARIANTS=A R1M_TOKENS=${RA_TOKENS:-131072,984000} \
  R1M_ENV="XE_BIN=$BIN LD_LIBRARY_PATH=$BIN STRATA_TP_SYNC_LOG=1 STRATA_TP_SYNC_CHECK=1 ${RA_ENV:-}" \
  bash bench/conc/r1m.sh
echo "--- read-ahead / evictions / growth (engine-r1mA.log)"
grep -a -E "read-ahead|gave .* back|KV replicas now|serve: prompt [0-9]{6} |serve: [0-9]+ tokens in|flex region|no room|no eviction" \
  logs/engine-r1mA.log | cut -c1-260
