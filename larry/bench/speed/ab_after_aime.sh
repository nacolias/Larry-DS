#!/bin/bash
# ab_after_aime.sh - waits for the AIME client (STOP file set: it ends after its current problem), then runs the
# short-prompt A/B (tab_all.sh: the service held + stopped, configs, the logits check, the service back).
cd /opt/ai/dsv41
while pgrep -f "[a]ime.py" > /dev/null; do sleep 10; done
echo "AIME client gone $(date +%T)"
TAB_PPL=1 bash bench/speed/tab_all.sh l0:STRATA_XE_CPU_LIGHT=0 l1:STRATA_XE_CPU_LIGHT=1 l2:STRATA_XE_CPU_LIGHT=2 \
  l4:STRATA_XE_CPU_LIGHT=4 l8:STRATA_XE_CPU_LIGHT=8 l0b:STRATA_XE_CPU_LIGHT=0
echo "A/B done $(date +%T)"
