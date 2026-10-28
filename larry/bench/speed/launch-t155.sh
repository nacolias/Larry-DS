#!/bin/bash
# launch-t155.sh - abwT155 (#3): the fused prompt attention v4 (block-read staging, the next chunk prefetched to L1; the window in place by
# position; compile-time phase skips): numerics + time at 4 positions (run_fsa.sh), then where its time goes (run_fsa2.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa.sh" "@bash bench/fa/run_fsa2.sh" > logs/abwT155.log 2>&1 < /dev/null &
echo "abwT155 launched: logs/abwT155.log"
