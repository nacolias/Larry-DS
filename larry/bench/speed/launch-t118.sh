#!/bin/bash
# launch-t118.sh - abwT118: card 0's op profile of p30k's second batch by shape and matrix (opprof-shapes.sh) - which of
# a span's ~8 GEMMs (2.4 ms a span, ~60 TFLOPS) and which norms / copies cost what
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh "@SKIP=42000 bash bench/speed/opprof-shapes.sh p30k.i32" > logs/abwT118.log 2>&1 < /dev/null &
echo "abwT118 launched: logs/abwT118.log"
