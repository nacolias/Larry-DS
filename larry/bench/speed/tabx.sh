#!/bin/bash
# tabx.sh NAME TESTS [VAR=value ...] (AB_ENGRAM=dirs: the engine's --engram) - one config of the 2026-10-24 afternoon A/B: the service's server + engine on 18081
# (the service must be stopped; AB_CTX = its --ctx) with extra environment, clocks pinned as the service pins them; 2 warm-up requests, then
# abx.py with TESTS (its flags, comma-separated: prompts / edit / dec) and agentx.py (agent); then the server stopped (SIGTERM to its session).
# Stops early when $AB_FLAG exists (the stand-in on :18080 saw a request).  Rows: bench/speed/out/abx-NAME.jsonl
set -u
cd /opt/ai/dsv41
N=$1; T=$2; shift 2
S=bench/speed
systemctl is-active -q larry-ds41 && { echo "the service is running - stop it first"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" "$@" <<'PY'
import json, sys
import os
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
if os.environ.get('AB_CTX'):   # the engine's context (--ctx) for this config
    c['args'][c['args'].index('--ctx') + 1] = os.environ['AB_CTX']
for kv in sys.argv[3:]:   # AB_ARGS=--flag:value:--flag2:value2 in the config's variables: more engine arguments
    if kv.startswith('AB_ARGS='):
        c['args'] += [a for a in kv.split('=', 1)[1].split(':') if a]
for kv in sys.argv[3:]:   # AB_ENGRAM=dir:dir in the config's variables: its --engram
    if kv.startswith('AB_ENGRAM='):
        c['args'][c['args'].index('--engram') + 1] = kv.split('=', 1)[1]
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
mkdir -p /run/ds41-test $S/out
rm -f logs/engine-$N.log $S/out/abx-$N.jsonl $S/out/agent-$N.jsonl
/usr/local/sbin/larry-gpu-clock pin > /dev/null
# (2026-10-26) the Engram tables through the filesystem as the service reads them - its user cannot open /dev/dm-0, root
# could, so every test server before abwT138 took the raw device path; a config sets STRATA_ENGRAM_RAWDEV to change it
env STRATA_ENGRAM_RAWDEV=0 "$@" RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
SP=$!
echo "[$N] server pid $SP, env: $*  $(date +%T)"
ok=0
for i in $(seq 1 90); do
  sleep 5
  kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; tail -5 logs/serve-$N.log; break; }
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)
  [ "$c" = "200" ] && { ok=1; break; }
  [ -n "${AB_FLAG:-}" ] && [ -e "$AB_FLAG" ] && break
done
if [ $ok = 1 ]; then
  echo "[$N] up $(date +%T)"
  PY=/opt/ai/dsv41/venv/bin/python
  taskset -c 15 $PY -I $S/speedtab.py --port 18081 --warm 2 | sed "s/^/[$N] /"
  F=""
  AG=0
  for t in ${T//,/ }; do
    case $t in
      prompts) F="$F --prompts ${AB_PROMPTS:-3000,4400,6000,8000}" ;;
      edit) F="$F --edit" ;;
      dec) F="$F --dec" ;;
      agent) AG=1 ;;
      needles) F="$F --needles ${AB_NEEDLES:-131072,262000}" ;;
    esac
  done
  [ -n "$F" ] && taskset -c 15 $PY -I $S/abx.py --port 18081 --label $N $F --log logs/engine-$N.log --out $S/out/abx-$N.jsonl \
    --stop-flag "${AB_FLAG:-/nonexistent}" 2>&1 | sed -u "s/^/[$N] /"
  [ $AG = 1 ] && taskset -c 15 $PY -I $S/agentx.py --port 18081 --label $N --log logs/engine-$N.log ${AGENT_ARGS:-} \
    --out $S/out/agent-$N.jsonl --stop-flag "${AB_FLAG:-/nonexistent}" 2>&1 | sed -u "s/^/[$N] /"
  grep -a -E "CPU-light|promotion cache|flex region|xe prefetch: layers" logs/engine-$N.log | tail -4 | sed "s/^/[$N] /"
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
