#!/bin/bash
# launch-dd1.sh - decode at depth, step 1 (2026-10-29): deeprun.sh dd1 in a window - agentd.py's deep agent session
# (thinking on, temperature 1, 64K -> ~80K) on a private server with STRATA_XE_CPU_LOG, for cpulog.py / pcsim.py
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/loop/deeprun.sh dd1" > logs/abw-dd1.log 2>&1 < /dev/null &
echo "launched: logs/abw-dd1.log"
