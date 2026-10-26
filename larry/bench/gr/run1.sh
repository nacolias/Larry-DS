#!/bin/bash
# run1.sh - the first measurement round after the build: tjob (JOBNT kernel bench), the hs_probe, tli2 (LI_XMX)
cd /opt/ai/dsv41/bench/gr
echo "== tjob $(date +%T)"; ./tjob.sh > tjob.log 2>&1; echo "tjob rc=$?"
echo "== tprobe $(date +%T)"; ./tprobe.sh > tprobe.out 2>&1; cat tprobe.out
for i in $(seq 1 120); do pgrep -x hs_probe > /dev/null || break; sleep 2; done
pgrep -x hs_probe > /dev/null && echo "hs_probe STILL RUNNING after 240 s"
echo "== tli2 $(date +%T)"; ./tli2.sh > tli2.log 2>&1; echo "tli2 rc=$?"
echo "ALL1 done $(date +%T)"
