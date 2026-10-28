#!/bin/bash
# launch-t152.sh - abwT152 (#3): the fused prompt attention v2 with each phase skipped (where its time goes);
# bench/fa/run_fsa2.sh
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa2.sh" > logs/abwT152.log 2>&1 < /dev/null &
echo "abwT152 launched: logs/abwT152.log"
