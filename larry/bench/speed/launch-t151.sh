#!/bin/bash
# launch-t151.sh - abwT151 (#3): the fused prompt attention v2 (block staging, pipelined loops) against the service's KGATHER +
# oneDNN SDPA path on the same inputs - numerics vs a double reference, time (bench/fa/run_fsa.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa.sh" > logs/abwT151.log 2>&1 < /dev/null &
echo "abwT151 launched: logs/abwT151.log"
