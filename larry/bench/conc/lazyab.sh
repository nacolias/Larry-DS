#!/bin/bash
# lazyab.sh - which engine-side factor makes the TP replicas of card 0's lazily mapped KV state differ from card 0
# (patch-tpsynclog.py's check: at a 131K conversation's first follow-up, rows in card 0's later KV allocation): on the
# staged engine (s2bin), a private server a config ("parallel": 2, STRATA_DENSE_LAZY_KV=1, KV_INIT 65536, the sync log
# and check): needle 200 (-> conversation slot 0), then a 131072-token needle + its deep decode + a 12K turn (abx.py ->
# slot 1); the "tp sync" lines.  Configs (LAZYAB_CONFIGS, default all): base, nopage (STRATA_TP_PAGE=0), nobar
# (STRATA_XE_BAR_STAGE=0).  An abw.sh "@" step.
cd /opt/ai/dsv41
for V in ${LAZYAB_CONFIGS:-base nopage nobar}; do
  N=lazy-$V
  CFG=$PWD/logs/cfg-$N.json
  python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
c['parallel'] = 2
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
  ENVV="XE_BIN=/opt/ai/dsv41/s2bin LD_LIBRARY_PATH=/opt/ai/dsv41/s2bin STRATA_DENSE_LAZY_KV=1 STRATA_DENSE_KV_INIT=65536 STRATA_TP_SYNC_LOG=1 STRATA_TP_SYNC_CHECK=1"
  [ $V = nopage ] && ENVV="$ENVV STRATA_TP_PAGE=0"
  [ $V = nobar ] && ENVV="$ENVV STRATA_XE_BAR_STAGE=0"
  rm -f logs/engine-$N.log
  T0=$(date '+%F %T')
  mkdir -p /run/ds41-test bench/engine1/p18081
  env $ENVV RUNTIME_DIRECTORY=/run/ds41-test DS41_PORT=18081 DS41_CONFIG=$CFG setsid ./ds41-serve.sh > logs/serve-$N.log 2>&1 < /dev/null &
  SP=$!
  echo "[$N] server pid $SP  $(date +%T)"
  ok=0
  for i in $(seq 1 90); do
    sleep 5
    kill -0 $SP 2>/dev/null || { echo "[$N] server exited"; break; }
    [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18081/health)" = "200" ] && { ok=1; break; }
  done
  if [ $ok = 1 ]; then
    sed 's/172\.23\.0\.1:18080/172.23.0.1:18081/g' needle.py > bench/engine1/p18081/needle.py
    echo "[$N] needle 200: $(python3 bench/engine1/p18081/needle.py 200 1 2>&1 | tail -1)"
    /opt/ai/dsv41/venv/bin/python -I bench/speed/abx.py --port 18081 --label $N --needles 131072 --log logs/engine-$N.log \
      --out bench/speed/out/abx-$N.jsonl 2>&1 | grep -E "needle|deep|turn" | cut -c1-200
  fi
  kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
  for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || break; }; sleep 2; done
  grep -a -E "tp sync|kv map:|KV replicas now" logs/engine-$N.log | grep -v MISMATCH | tail -12 | cut -c1-200
  grep -a -c "MISMATCH" logs/engine-$N.log | sed "s/^/[$N] mismatch lines: /"
  f=$(journalctl -k --since "$T0" --no-pager | grep -c "Faulted Address")
  echo "[$N] faults in the kernel log since its start: $f"
  [ "$f" -gt 0 ] && { bash /opt/ai/dsv41/logs/gtreset.sh; break; }
done
