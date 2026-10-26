#!/bin/bash
# ab5.sh - one window after the AIME client's current problem (STOP file): the 3-4K-prompt A/B (ab4's: pp3500 / pp2048
# with the prefetch from 2048 everywhere / the service's 4096 + 2048 / none), the standard speed table again
# (run_all2.sh: DSpark and plain at the service's settings, labels dspark2 / plain2), tsvc2 on the restored service,
# then AIME again.  Each part holds, stops and restores the service itself.
cd /opt/ai/dsv41
while pgrep -f "[a]ime.py" > /dev/null; do sleep 10; done
echo "AIME client gone $(date +%T)"
TAB_SPECS="3500:0 2048:0" TAB_REPS=5 bash bench/speed/tab_all.sh q1:STRATA_XE_PREFETCH_MIN=2048 q0 \
  q2:STRATA_XE_PREFETCH_MIN=8192,STRATA_XE_PREFETCH_MIN_B=8192
echo "A/B done $(date +%T)"
bash bench/speed/run_all2.sh
echo "speed table done $(date +%T)"
if curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health | grep -q 200; then
  echo "tsvc2 $(date +%T)"
  timeout 1200 bash tsvc2.sh 2>&1 | tail -16 | cut -c1-200
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
