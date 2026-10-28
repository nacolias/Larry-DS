#!/bin/bash
# launch-t150.sh - abwT150 (#3): the fused prompt attention (bench/fa/fsa_kernel.inc) against the service's KGATHER +
# oneDNN SDPA path on the same inputs - numerics vs a double reference, time (bench/fa/run_fsa.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/run_fsa.sh" > logs/abwT150.log 2>&1 < /dev/null &
echo "abwT150 launched: logs/abwT150.log"
