#!/bin/bash
# launch-t105.sh - abwT108: card 0's ggml-sycl time op by op on a fresh 20K prompt (bench/speed/opprof.sh)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh "@bash bench/speed/opprof.sh p20k.i32" > logs/abwT108.log 2>&1 < /dev/null &
echo "abwT108 launched: logs/abwT108.log"
