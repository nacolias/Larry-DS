#!/bin/bash
# vtest.sh NAME - one server with a "vision" section (the service's config + the encoder; port 18081, the service stopped,
# as tabx.sh) and vtest.py's picture checks; then the server stopped.  Logs: logs/serve-NAME.log, logs/engine-NAME.log.
set -u
cd /opt/ai/dsv41
N=${1:-vis}
systemctl is-active -q larry-ds41 && { echo "the service is running - stop it first"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
CFG=$PWD/logs/cfg-$N.json
python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
c['vision'] = {"exe": "/opt/ai/dsv41/ds41-vision.sh", "mmproj": "/opt/ai/dsv41/models/gsq1017/V41-MTP-Vision-1017.Q2_K.gguf",
               "model": "/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf", "threads": 6}
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
rm -f logs/engine-$N.log
/usr/local/sbin/larry-gpu-clock pin > /dev/null
env STRATA_ENGRAM_RAWDEV=0 RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
SP=$!
echo "[$N] server pid $SP $(date +%T)"
ok=0
for i in $(seq 1 90); do
  sleep 5
  kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; tail -20 logs/serve-$N.log; break; }
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)
  [ "$c" = "200" ] && { ok=1; break; }
done
if [ $ok = 1 ]; then
  echo "[$N] up $(date +%T)"
  KEY=$(grep ^VLLM_API_KEY /opt/docker/vllm/.env | cut -d= -f2-)
  VTEST_LOG=logs/engine-$N.log VTEST_KEY="$KEY" timeout 1800 /opt/ai/dsv41/tvenv/bin/python -I bench/vision/vtest.py 18081 logs/vtest-$N 2>&1 | sed "s/^/[$N] /"
  grep -a -E "vision|image|GENI|ERR" logs/serve-$N.log | tail -8 | sed "s/^/[$N] serve: /"
fi
kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
for i in $(seq 1 90); do
  pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || { ss -ltn | grep -q ":18081 " || break; }; }
  sleep 2
done
pgrep -x ds41_generate_x > /dev/null && echo "[$N] ENGINE STILL RUNNING after 180 s"
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "[$N] done $(date +%T)"
