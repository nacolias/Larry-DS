#!/bin/bash
# prof1m.sh [TOKENS] [BIN] - card 0's ggml-sycl time op by op over a long prompt: a private server (the service's config,
# port 18081, GGML_SYCL_OP_PROF=1 - a cumulative dump every PROF_EVERY marks - and STRATA_BATCH_TRACE=1 between them)
# reads one needle of TOKENS (default 984000) from empty, then its 400-token decode and 12K turn (abx.py).
# bench/longctx/profdiff.py turns the dumps into a table an interval (the batches each one spans).  BIN: a staged engine.
# An abw.sh "@" step.
cd /opt/ai/dsv41
T=${1:-984000}
BIN=${2:-}
N=prof1m
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
ENVV="GGML_SYCL_OP_PROF=1 GGML_SYCL_OP_PROF_EVERY=${PROF_EVERY:-250000} GGML_SYCL_OP_PROF_TOP=200 STRATA_BATCH_TRACE=1"
[ -n "$BIN" ] && ENVV="$ENVV XE_BIN=$BIN LD_LIBRARY_PATH=$BIN"
rm -f logs/engine-$N.log
mkdir -p /run/ds41-test
env $ENVV RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
SP=$!
echo "[$N] server pid $SP, env: $ENVV  $(date +%T)"
ok=0
for i in $(seq 1 90); do
  sleep 5
  kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; break; }
  [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)" = "200" ] && { ok=1; break; }
done
if [ $ok = 1 ]; then
  /opt/ai/dsv41/venv/bin/python -I bench/speed/abx.py --port 18081 --label $N --needles "$T" --log logs/engine-$N.log \
    --out bench/speed/out/abx-$N.jsonl 2>&1 | grep -E "needle|deep|turn" | cut -c1-200
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || break; }; sleep 2; done
python3 -I bench/longctx/profdiff.py logs/engine-$N.log | head -${PROF_LINES:-400}
