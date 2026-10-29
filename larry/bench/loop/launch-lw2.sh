#!/bin/bash
# launch-lw2.sh - the loop guard into the service (patch-loopwrap.py + svc-loopwrap.py; launch-lw.sh's window ended before
# it: a second window's idle check had tripped its stand-in), then decode at depth, step 1 (deeprun.sh dd1: agentd.py's
# deep agent session with STRATA_XE_CPU_LOG, the guard in its config); the restart at the window's end loads the guard;
# then post-lw.sh lw2: tsvc2 + concurtest + loopprobe.py (logs/tsvc2-lw2.log)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@python3 patches/patch-loopwrap.py && python3 patches/svc-loopwrap.py" \
  "@bash bench/loop/deeprun.sh dd1" > logs/abw-lw2.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh lw2 > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-lw2.log, then logs/tsvc2-lw2.log"
