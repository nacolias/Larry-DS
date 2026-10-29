#!/bin/bash
# post-lw.sh [NAME] - after launch-NAME.sh's window (default lw): waits for it to end (logs/abw-NAME.log's "window
# end", up to 9 h), then the service checks on what it runs: tsvc2.sh, concurtest, loopprobe.py and the server's
# wrap-up lines.  Log: logs/tsvc2-NAME.log.  (Never start a second abw.sh while a window holds the service: its idle
# check GETs /metrics from the stand-in, which ends the running window - 2026-10-29 17:41.)
cd /opt/ai/dsv41
N=${1:-lw}
W=logs/abw-$N.log
O=logs/tsvc2-$N.log
for i in $(seq 1 1080); do
  grep -q "window end" $W 2>/dev/null && break
  grep -q -E "hold kept|nothing run" $W 2>/dev/null && { echo "the window did not end normally" > $O; exit 1; }
  sleep 30
done
grep -q "service health 200" $W || { echo "the service is not healthy after the window" > $O; exit 1; }
{
  echo "config: $(grep -c '"reasoning_loop_recovery": "wrap"' larry-ds41.json) wrap line, server patch: $(grep -c patch-loopwrap.py xolotl/serve/server.py) mentions"
  md5sum xolotl/build-xe/ds41_generate_xe xolotl/build-xe/libds41_tp.so xolotl/build-xe/libds41_xe.so | cut -c1-12,34-
  bash tsvc2.sh
  echo "--- concurtest"
  PYTHONPATH=/opt/ai/dsv41 DS41_URL=http://172.23.0.1:18080 DS41_KEY_FILE=/opt/docker/vllm/.env python3 bench/conc/concurtest.py logs/engine.log 2>&1 | tail -15
  echo "--- loop probe"
  T0=$(date '+%F %T')
  DS41_URL=http://172.23.0.1:18080 DS41_KEY_FILE=/opt/docker/vllm/.env python3 bench/loop/loopprobe.py 2>&1 | tail -3
  DS41_URL=http://172.23.0.1:18080 DS41_KEY_FILE=/opt/docker/vllm/.env python3 bench/loop/loopprobe.py --answer 2>&1 | tail -3
  journalctl -u larry-ds41 --since "$T0" --no-pager -o cat | grep -E "reasoning_loop_recovery|thinking budget reached|answer repeated" | cut -c1-200
  echo "--- done $(date '+%F %T')"
} > $O 2>&1
