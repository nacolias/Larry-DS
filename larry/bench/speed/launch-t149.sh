#!/bin/bash
# launch-t149.sh - abwT149 (#3): the inner loops again, + operands from global memory (2D block loads) (bench/fa/mqa_mb.cpp)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_mb.sh" > logs/abwT149.log 2>&1 < /dev/null &
echo "abwT149 launched: logs/abwT149.log"
