#!/bin/bash
# ab7.sh - after ab6.sh's A/B (tab_all.sh d0 / dn, left running when ab6.sh was stopped): DSpark's long prompts from
# empty (6.7K, 31K, 97K: pp512's first messages at depths 8192 / 32768 / 98304, the speed table's order) with STRATA_XE_CPU_LIGHT 0 / 4 - the speed table had
# them 2 / 10 s slower with the pool's light experts on (dense phase only; plain unchanged).  Then tsvc2, AIME.
cd /opt/ai/dsv41
while pgrep -f "[t]ab_all.sh" > /dev/null; do sleep 10; done
echo "ab6 A/B done $(date +%T)"
TAB_SPECS="512:8192,32768,98304" TAB_REPS=3 bash bench/speed/tab_all.sh z0:STRATA_XE_CPU_LIGHT=0 z4
echo "A/B done $(date +%T)"
if curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health | grep -q 200; then
  echo "tsvc2 $(date +%T)"
  timeout 1200 bash tsvc2.sh 2>&1 | tail -16 | cut -c1-200
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
