#!/bin/bash
# after-lw.sh - launch-dd1.sh once post-lw.sh's service checks are done (logs/tsvc2-lw.log "--- done"), so its window
# cannot take the service from under them (abw.sh's idle gate then waits 20 min after their last request); gives up
# after 3 h or when post-lw.sh failed.
cd /opt/ai/dsv41
for i in $(seq 1 360); do
  grep -q -- "--- done" logs/tsvc2-lw.log 2>/dev/null && break
  grep -q -E "did not end normally|not healthy" logs/tsvc2-lw.log 2>/dev/null && exit 1
  sleep 30
done
grep -q -- "--- done" logs/tsvc2-lw.log || exit 1
bash bench/loop/launch-dd1.sh
