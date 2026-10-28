#!/bin/bash
# tsvc2-ra.sh [NAME] - the service checks after an install: tsvc2.sh, then bench/conc/concurtest.py on the service's log.
# Waits for the window that installed it to end first (abw.sh's "window end").  Log: logs/tsvc2-NAME.log.
cd /opt/ai/dsv41
N=${1:-ra}
for i in $(seq 1 120); do grep -q "window end" logs/abw-${N}inst.log 2>/dev/null && break; sleep 5; done
{
  md5sum xolotl/build-xe/ds41_generate_xe xolotl/build-xe/libds41_tp.so xolotl/build-xe/libds41_xe.so | cut -c1-12,34-
  bash tsvc2.sh
  echo "--- concurtest"
  DS41_URL=http://172.23.0.1:18080 DS41_KEY_FILE=/opt/docker/vllm/.env python3 bench/conc/concurtest.py logs/engine.log 2>&1 | tail -15
} > logs/tsvc2-$N.log 2>&1
