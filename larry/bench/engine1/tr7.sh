#!/bin/bash
# tr7.sh - engine1 round 8, validation: rebuild libds41_tp with the new defaults (route_jobs a lane a pick, the expert
# kernels' job bound / grid 8, tuned mat-vec forms with exact XMX), then the service's test sequence twice on a private
# port (tpg.sh: needles, svcbench, tool call).  Service held.
cd /opt/ai/dsv41/bench/engine1
TARGETS="ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
TPG_ROUNDS=2 bench/engine1/tpg.sh kern 2>&1
echo "TR7 done $(date +%T)"
