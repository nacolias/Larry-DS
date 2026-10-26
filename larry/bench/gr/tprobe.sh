#!/bin/bash
# tprobe.sh - build and run hs_probe (one expert card, STRATA_HS_DEV=1; ~10-20 s, bounded spins, a watchdog, no
# card-to-card copies), detached from the ssh session (setsid / nohup) with its log next to it (hs_probe.log).
# The service must be stopped (one GPU tenant).  Poll: grep -c "^done" hs_probe.log
cd "$(dirname "$(readlink -f "$0")")"
source /opt/ai/dsv41/xe-env.sh
icpx -fsycl -O2 -std=c++20 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" hs_probe.cpp -o hs_probe -lze_loader > hs_probe.build.log 2>&1 \
  || { echo "build failed"; tail -20 hs_probe.build.log; exit 1; }
setsid nohup ./hs_probe > hs_probe.log 2>&1 < /dev/null &
echo "started pid $!"
