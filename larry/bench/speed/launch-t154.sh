#!/bin/bash
# launch-t154.sh - abwT154 (#3): the fused prompt attention v3 (the window keys read in place from an fp16 copy by
# position; compile-time phase skips): numerics + time at 4 positions (run_fsa.sh), then where its time goes (run_fsa2.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa.sh" "@bash bench/fa/run_fsa2.sh" > logs/abwT154.log 2>&1 < /dev/null &
echo "abwT154 launched: logs/abwT154.log"
