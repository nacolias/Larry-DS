#!/bin/bash
# r1m.sh - the 1M crash of s1b (2026-10-27 18:18: after a 970K needle in conversation slot 1, the next request's first
# TP pass faulted every card's compute engine) reproduced with STRATA_ADDR_LOG=1, in two variants, each on a private
# server with "parallel": 2: A = card 0's KV lazily mapped (STRATA_DENSE_LAZY_KV=1, KV_INIT 65536), B = card 0's KV
# allocated at load.  Each: needle 200 (-> conversation slot 0), then a long needle (R1M_TOKENS, default 600000) + its
# deep decode + a 12K turn (abx.py; -> slot 1).  A fault: its addresses matched against the engine log's "addr:" ranges,
# then a GT reset of every card (bench/../logs/gtreset.sh) before the next variant.  An abw.sh "@" step.
# R1M_VARIANTS (default "A B"), R1M_TOKENS (the needles, comma-separated, default 600000), R1M_ENV (more environment for
# both servers, e.g. a staged engine's XE_BIN / LD_LIBRARY_PATH and diagnostics).
cd /opt/ai/dsv41
for V in ${R1M_VARIANTS:-A B}; do
  N=r1m$V
  CFG=$PWD/logs/cfg-$N.json
  python3 -I - "$CFG" "$N" <<'PY'
import json, sys
c = json.load(open('/opt/ai/dsv41/larry-ds41.json'))
c['log'] = '/opt/ai/dsv41/logs/engine-%s.log' % sys.argv[2]
c['port'] = 18081
c['parallel'] = 2
json.dump(c, open(sys.argv[1], 'w'), indent=2)
PY
  ENVV="STRATA_ADDR_LOG=1"
  [ $V = A ] && ENVV="$ENVV STRATA_DENSE_LAZY_KV=1 STRATA_DENSE_KV_INIT=65536"
  ENVV="$ENVV ${R1M_ENV:-}"
  rm -f logs/engine-$N.log
  T0=$(date '+%F %T')
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
    sed 's/172\.23\.0\.1:18080/172.23.0.1:18081/g' needle.py > bench/engine1/p18081/needle.py
    echo "[$N] needle 200: $(python3 bench/engine1/p18081/needle.py 200 1 2>&1 | tail -1)"
    /opt/ai/dsv41/venv/bin/python -I bench/speed/abx.py --port 18081 --label $N --needles "${R1M_TOKENS:-600000}" --log logs/engine-$N.log \
      --out bench/speed/out/abx-$N.jsonl 2>&1 | grep -E "needle|deep|turn" | cut -c1-200
  fi
  kill -TERM -$SP 2>/dev/null || kill -TERM $SP 2>/dev/null
  for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || { kill -0 $SP 2>/dev/null || break; }; sleep 2; done
  grep -a -E "serve: prompt [0-9]{6} |decode failed|DEVICE_LOST|KV replicas now|kv map:|gave .* back|tp sync|tp prof" logs/engine-$N.log | tail -40 | cut -c1-230
  f=$(journalctl -k --since "$T0" --no-pager | grep -c "Faulted Address")
  echo "[$N] faults in the kernel log since its start: $f"
  if [ "$f" -gt 0 ]; then
    journalctl -k --since "$T0" --no-pager > logs/r1m-$V-dmesg.txt
    python3 -I - logs/r1m-$V-dmesg.txt logs/engine-$N.log <<'PY'
import re, sys
dm = open(sys.argv[1], errors="replace").read()
faults = []
for m in re.finditer(r"xe (\S+): \[drm\].*?Faulted Address: 0x([0-9a-f]+).*?EngineClass: (\d+ \w+)", dm, re.S):
    a = int(m.group(2), 16)
    if a & (1 << 47): a |= 0xffff000000000000   # xe prints the VA without its sign extension
    faults.append((m.group(1), a, m.group(3)))
rng = []
for ln in open(sys.argv[2], errors="replace"):
    m = re.search(r"addr: (vmm \w+|\S+.*?) (0x[0-9a-f]+) \+0x([0-9a-f]+)", ln)
    if m:
        rng.append((int(m.group(2), 16), int(m.group(3), 16), ln.strip()[:140]))
seen = set()
for dev, a, eng in faults:
    if (dev, a) in seen: continue
    seen.add((dev, a))
    hits = [r for r in rng if r[0] <= a < r[0] + r[1]]
    print("fault %s %s 0x%x: %s" % (dev, eng, a, hits[-1][2] if hits else "in no logged range"))
    for r in hits[-3:-1]: print("     also " + r[2])
PY
    bash /opt/ai/dsv41/logs/gtreset.sh
  fi
done
