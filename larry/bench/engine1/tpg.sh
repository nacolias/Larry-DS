#!/bin/bash
# tpg.sh NAME [VAR=value ...] - the service's server + engine on a private port (18081; the service must be stopped)
# with extra environment, then the tsvc2 sequence that broke with the pager on (needle 200, svcbench, needles 8K x2 /
# 16K / 31K, the tool call) with the answers shown, then the server stopped (SIGTERM, as systemd does)
set -u
cd /opt/ai/dsv41
N=$1; shift
systemctl is-active -q larry-ds41 && { echo "the service is running - stop it first"; exit 1; }
pgrep -f "[d]s41_generate_xe" > /dev/null && { echo "an engine is running"; exit 1; }
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
P=bench/engine1/p18081
mkdir -p $P /run/ds41-test
for f in needle.py svcbench.py tooltest.py; do sed 's/172\.23\.0\.1:18080/172.23.0.1:18081/g' $f > $P/$f; done
rm -f logs/engine-$N.log
/usr/local/sbin/larry-gpu-clock pin > /dev/null
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
  echo "[$N] up $(date +%T); engine pid $(pgrep -f '[d]s41_generate_xe' | head -1), STRATA_TP_PAGE=$(tr '\0' '\n' < /proc/$(pgrep -f '[d]s41_generate_xe' | head -1)/environ | grep '^STRATA_TP_PAGE=' | cut -d= -f2)"
  ans() { python3 -c 'import sys; L = sys.stdin.read().splitlines(); print((L[-1] if L else "?"), "|", (L[0][:100] if L else ""))'; }
  for rnd in $(seq 1 ${TPG_ROUNDS:-1}); do
    echo "-- round $rnd"
    echo "needle 200: $(python3 $P/needle.py 200 1 2>&1 | ans)"
    python3 $P/svcbench.py 2>&1 | cut -c1-150
    for n in 8000 8000 16000 31000; do echo "needle $n: $(python3 $P/needle.py $n 1 2>&1 | ans)"; done
    echo "tool: $(python3 $P/tooltest.py 2>&1 | tail -2 | head -1 | cut -c1-120)"
  done
  grep -a -E "dense card 0 seeded|tp pager: on|dense card's expert cap" logs/engine-$N.log | cut -c1-150 | tail -4
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null   # the session (setsid): server + engine, as systemd's stop
for i in $(seq 1 90); do
  pgrep -f "[d]s41_generate_xe" > /dev/null || { kill -0 $SP 2>/dev/null || { ss -ltn | grep -q ":18081 " || break; }; }
  sleep 2
done
pgrep -f "[d]s41_generate_xe" > /dev/null && echo "[$N] ENGINE STILL RUNNING after 180 s"
kill -0 $SP 2>/dev/null && echo "[$N] SERVER STILL RUNNING after 180 s"
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "[$N] done $(date +%T)"
