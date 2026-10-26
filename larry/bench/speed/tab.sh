#!/bin/bash
# tab.sh NAME [VAR=value ...] - one config of the short-prompt A/B: the service's server + engine on 18081 (the
# service must be stopped) with extra environment, clocks pinned as the service pins them; 3 warm-up requests, then
# speedtab.py: pp512 at depths 0 and 8192, pp160 at depth 0 (TAB_SPECS; tg 32, TAB_REPS runs each, default 5); then the server
# stopped (SIGTERM to its session).  Same launch as tspeed.sh.  Results: bench/speed/out/tab-NAME-{512,160}.json
set -u
cd /opt/ai/dsv41
N=$1; shift
S=bench/speed
systemctl is-active -q larry-ds41 && { echo "the service is running - stop it first"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
mkdir -p /run/ds41-test $S/out
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
  E=$(pgrep -x ds41_generate_x | head -1)
  echo "[$N] up $(date +%T); engine pid $E, STRATA_SERVE_DSPARK='$(tr '\0' '\n' < /proc/$E/environ | grep '^STRATA_SERVE_DSPARK=' | cut -d= -f2-)'"
  PY=/opt/ai/dsv41/venv/bin/python
  taskset -c 15 $PY -I $S/speedtab.py --port 18081 --warm 3
  for spec in ${TAB_SPECS:-512:0,8192 160:0}; do   # PP:DEPTHS ...
    pp=${spec%%:*}
    echo "[$N] pp$pp at depths ${spec#*:} $(date +%T)"
    taskset -c 15 $PY -I $S/speedtab.py --port 18081 --label $N-$pp --pp $pp --tg 32 --depths ${spec#*:} \
      --reps ${TAB_REPS:-5} --log logs/engine-$N.log --out $S/out/tab-$N-$pp.json
  done
  grep -a "CPU-light" logs/engine-$N.log | tail -1
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null   # the session (setsid): server + engine, as systemd's stop
for i in $(seq 1 90); do
  pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || { ss -ltn | grep -q ":18081 " || break; }; }
  sleep 2
done
pgrep -x ds41_generate_x > /dev/null && echo "[$N] ENGINE STILL RUNNING after 180 s"
kill -0 $SP 2>/dev/null && echo "[$N] SERVER STILL RUNNING after 180 s"
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "[$N] done $(date +%T)"
