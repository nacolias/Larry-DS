#!/bin/bash
# ab4.sh - 3-4K-token prompts with the prefetch range CL..39 (commit after 8c430ac5): pp3500 / pp2048 at depth 0 with
# the prefetch from 2048 everywhere (q1: a 3.5K prompt prefetches layers 0-39 - the service before), the service's
# 4096 / 2048 (q0: layers 20-39), none (q2).  Waits for the AIME client (STOP file), the A/B (the service held, then
# back), tsvc2 on the restored service (the new binary), AIME again.
cd /opt/ai/dsv41
while pgrep -f "[a]ime.py" > /dev/null; do sleep 10; done
echo "AIME client gone $(date +%T)"
TAB_SPECS="3500:0 2048:0" TAB_REPS=5 bash bench/speed/tab_all.sh q1:STRATA_XE_PREFETCH_MIN=2048 q0 \
  q2:STRATA_XE_PREFETCH_MIN=8192,STRATA_XE_PREFETCH_MIN_B=8192
echo "A/B done $(date +%T)"
if curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health | grep -q 200; then
  echo "tsvc2 $(date +%T)"
  timeout 1200 bash tsvc2.sh 2>&1 | tail -16 | cut -c1-200
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
