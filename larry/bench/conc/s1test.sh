#!/bin/bash
# s1test.sh NAME [VAR=value ...] - concurrency stage 1 on a private server (port 18081, the service stopped - an abw.sh
# "@" step): needle 200, seqtest.py (two resident conversations), reusetest.py (the pool beside them), the tool call,
# needle 8K, then the server stopped (SIGTERM, as systemd does).  Log: logs/engine-NAME.log
# CONC_PARALLEL=N (the script's own environment): the config gets "parallel": N (the server's batch client: --batch N)
# and concurtest.py (requests at once) runs first.  CONC_NEEDLES=a,b: long needles + their turns (abx.py) at the end.
set -u
cd /opt/ai/dsv41
N=$1; shift
systemctl is-active -q larry-ds41 && { echo "the service is running - stop it first"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
import os
if os.environ.get('CONC_PARALLEL'):
    c['parallel'] = int(os.environ['CONC_PARALLEL'])
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
P=bench/engine1/p18081
mkdir -p $P /run/ds41-test
for f in needle.py tooltest.py; do sed 's/172\.23\.0\.1:18080/172.23.0.1:18081/g' $f > $P/$f; done
rm -f logs/engine-$N.log
env "$@" RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
SP=$!
echo "[$N] server pid $SP, env: $*  $(date +%T)"
ok=0
for i in $(seq 1 90); do
  sleep 5
  kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; tail -5 logs/serve-$N.log; break; }
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)
  [ "$c" = "200" ] && { ok=1; break; }
done
if [ $ok = 1 ]; then
  echo "[$N] up $(date +%T)"
  grep -a -E "conversation slots|tp: [0-9]+ pass graphs|dense card 0 seeded|free GiB" logs/engine-$N.log | head -5 | cut -c1-160
  ans() { python3 -c 'import sys; L = sys.stdin.read().splitlines(); print((L[-1] if L else "?"), "|", (L[0][:100] if L else ""))'; }
  echo "needle 200: $(python3 $P/needle.py 200 1 2>&1 | ans)"
  export DS41_URL=http://172.23.0.1:18081 DS41_KEY_FILE=/opt/docker/vllm/.env PYTHONPATH=/opt/ai/dsv41
  [ -n "${CONC_PARALLEL:-}" ] && python3 bench/conc/concurtest.py logs/engine-$N.log 2>&1
  python3 bench/conc/seqtest.py logs/engine-$N.log 2>&1
  python3 bench/conc/reusetest.py logs/engine-$N.log 2>&1 | grep -E "turn|E,|side|RESULT|engine log"
  echo "tool: $(python3 $P/tooltest.py 2>&1 | tail -2 | head -1 | cut -c1-120)"
  echo "needle 8000: $(python3 $P/needle.py 8000 1 2>&1 | ans)"
  if [ -n "${CONC_NEEDLES:-}" ]; then
    /opt/ai/dsv41/venv/bin/python -I bench/speed/abx.py --port 18081 --label $N --needles "$CONC_NEEDLES" --log logs/engine-$N.log \
      --out bench/speed/out/abx-$N.jsonl 2>&1 | grep -E "needle|deep|turn" | cut -c1-200
  fi
  grep -a -E "kv map|KV replicas now|batch slots|slot [0-9]+ \(conversation" logs/engine-$N.log | tail -12 | cut -c1-170
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do
  pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || { ss -ltn | grep -q ":18081 " || break; }; }
  sleep 2
done
pgrep -x ds41_generate_x > /dev/null && echo "[$N] ENGINE STILL RUNNING after 180 s"
echo "[$N] done $(date +%T)"
