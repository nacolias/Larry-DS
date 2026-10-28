#!/bin/bash
# launch-t148.sh - abwT148 (#3): the fused sparse attention's inner loops on the B70 (bench/fa/mqa_mb.cpp)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_mb.sh" > logs/abwT148.log 2>&1 < /dev/null &
echo "abwT148 launched: logs/abwT148.log"
