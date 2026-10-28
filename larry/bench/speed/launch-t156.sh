#!/bin/bash
# launch-t156.sh - abwT156 (#3): the fused prompt attention v5 (block-IO Q conversion, 2 rows in flight, P single-buffered; the window in place by
# position; compile-time phase skips): numerics + time at 4 positions (run_fsa.sh), then where its time goes (run_fsa2.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa.sh" "@bash bench/fa/run_fsa2.sh" > logs/abwT156.log 2>&1 < /dev/null &
echo "abwT156 launched: logs/abwT156.log"
