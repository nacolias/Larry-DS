#!/bin/bash
# wb.sh - the stage 2 measurement (patch-widthbench.py) on a private server (port 18081; an abw.sh "@" step): an essay,
# an 8K needle, another essay, a 31K needle - each followed by STRATA_TP_WIDTH_BENCH passes of widths 1..7 (own rows /
# random rows / two conversations' rows: the previous request's text the other conversation), then the server stopped.
# The engine log's "width bench" and "DSpark cycle" lines.
cd /opt/ai/dsv41
N=wb
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
c.pop('parallel', None)
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
rm -f logs/engine-$N.log
mkdir -p /run/ds41-test bench/engine1/p18081
sed 's/172\.23\.0\.1:18080/172.23.0.1:18081/g' needle.py > bench/engine1/p18081/needle.py
env STRATA_TP_WIDTH_BENCH=${WB_REPS:-9} RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
SP=$!
ok=0
for i in $(seq 1 90); do
  sleep 5
  kill -0 $SP 2>/dev/null || { echo "server exited"; break; }
  [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)" = "200" ] && { ok=1; break; }
done
if [ $ok = 1 ]; then
  export DS41_URL=http://172.23.0.1:18081 PYTHONPATH=/opt/ai/dsv41 DS41_KEY_FILE=/opt/docker/vllm/.env
  echo "essay 0: $(python3 bench/conc/wbessay.py 0 400 2>&1 | tail -1)"
  echo "needle 8000: $(python3 bench/engine1/p18081/needle.py 8000 1 2>&1 | tail -1)"
  echo "essay 1: $(python3 bench/conc/wbessay.py 1 400 2>&1 | tail -1)"
  echo "needle 31000: $(python3 bench/engine1/p18081/needle.py 31000 1 2>&1 | tail -1)"
  for i in $(seq 1 30); do [ "$(grep -ac "width bench at" logs/engine-$N.log)" -ge 4 ] && break; sleep 2; done   # the last bench runs after its answer
  grep -a "width bench\|DSpark cycle" logs/engine-$N.log
  grep -a "DSpark\|draft" logs/engine-$N.log | grep -a -i "ms\b\|tokens/pass" | tail -3 | cut -c1-160
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || break; }; sleep 2; done
