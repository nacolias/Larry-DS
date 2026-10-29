#!/bin/bash
# launch-lw.sh - the 2026-10-29 window: ltkl.sh lk1 (long-context numerics of the 10-28 prompt-path changes, after the
# morning's thinking loops), then the loop guard into the service (patch-loopwrap.py + svc-loopwrap.py; the restart at
# the window's end loads them); after the window post-lw.sh: tsvc2 + concurtest + loopprobe.py (logs/tsvc2-lw.log)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/loop/ltkl.sh lk1" \
  "@python3 patches/patch-loopwrap.py && python3 patches/svc-loopwrap.py" > logs/abw-lw.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-lw.log, then logs/tsvc2-lw.log"
