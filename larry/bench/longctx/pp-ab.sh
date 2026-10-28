#!/bin/bash
# pp-ab.sh BIN TOKENS NAME:VAR=v,VAR=v ... - long prompts on a staged engine (BIN) under each configuration in turn: a
# private server (the service's config on :18081), one fresh needle of TOKENS (bench/speed/abx.py: then its 400-token
# decode and a 12K turn), the engine's prompt phases and checks; engine logs logs/engine-ppab-NAME.log.  New CAT errors /
# faults after a configuration: a GT reset and stop.  An abw.sh "@" step.
cd /opt/ai/dsv41
BIN=$1; T=$2; shift 2
for CFG in "$@"; do
  N=${CFG%%:*}; E=${CFG#*:}; [ "$E" = "$CFG" ] && E=
  CF=$PWD/logs/cfg-ppab.json
  python3 -I - "$CF" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-ppab-%s.log' % sys.argv[2]
c['port'] = 18081
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
  ENVV="XE_BIN=$BIN LD_LIBRARY_PATH=$BIN ${E//,/ }"
  rm -f logs/engine-ppab-$N.log
  T0=$(date '+%F %T')
  mkdir -p /run/ds41-test
  env $ENVV RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CF setsid ./ds41-serve.sh > logs/serve-ppab-$N.log 2>&1 < /dev/null &
  SP=$!
  echo "[$N] server pid $SP, env: $ENVV  $(date +%T)"
  ok=0
  for i in $(seq 1 90); do
    sleep 5
    kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; break; }
    [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)" = "200" ] && { ok=1; break; }
  done
  if [ $ok = 1 ]; then
    /opt/ai/dsv41/venv/bin/python -I bench/speed/abx.py --port 18081 --label ppab-$N --needles "$T" --log logs/engine-ppab-$N.log \
      --out bench/speed/out/abx-ppab-$N.jsonl 2>&1 | grep -E "needle|deep|turn" | cut -c1-220
  fi
  kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
  for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || break; }; sleep 2; done
  grep -a -E "larry li|prompt phases: [0-9]{6}|decode failed|DEVICE_LOST|OUT_OF" logs/engine-ppab-$N.log | cut -c1-220 | tail -14
  f=$(journalctl -k --since "$T0" --no-pager | grep -c -i "Faulted Address\|cat error")
  echo "[$N] faults / CAT errors in the kernel log since its start: $f"
  if [ "$f" -gt 0 ]; then bash /opt/ai/dsv41/logs/gtreset.sh; echo "[$N] stopped after a fault"; exit 1; fi
done
