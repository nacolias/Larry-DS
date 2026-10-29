#!/bin/bash
# deeprun.sh NAME [VAR=value ...] - decode at depth: a private server with the service's config on port 18081 (the
# service stopped - an abw.sh "@" step) and STRATA_XE_CPU_LOG=logs/cpulog-NAME.txt (the CPU pool's experts and the
# promotion cache's events per pass: bench/engine1/cpulog.py, pcsim.py), agentd.py's deep agent session (AGENTD_ARGS,
# default --first 64000 --turns 24: thinking on, temperature 1), then the server stopped (SIGTERM, as systemd does).
# Logs: logs/engine-NAME.log, logs/serve-NAME.log, bench/loop/out/agentd-NAME.jsonl.
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
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
mkdir -p /run/ds41-test bench/loop/out
rm -f logs/engine-$N.log logs/cpulog-$N.txt bench/loop/out/agentd-$N.jsonl
env "$@" STRATA_XE_CPU_LOG=$PWD/logs/cpulog-$N.txt RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
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
  /opt/ai/dsv41/venv/bin/python -I bench/loop/agentd.py --label $N --log logs/engine-$N.log \
    --out bench/loop/out/agentd-$N.jsonl ${AGENTD_ARGS:---first 64000 --turns 24} --stop-flag ${AB_FLAG:-/nonexistent} 2>&1
  grep -a -h "promotion cache:" logs/engine-$N.log | tail -2 | cut -c1-200
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do
  pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || { ss -ltn | grep -q ":18081 " || break; }; }
  sleep 2
done
pgrep -x ds41_generate_x > /dev/null && echo "[$N] ENGINE STILL RUNNING after 180 s"
grep -a -h "promotion cache:" logs/engine-$N.log | tail -1 | cut -c1-200
echo "[$N] done $(date +%T); CPU log $(wc -l < logs/cpulog-$N.txt 2>/dev/null) lines"
