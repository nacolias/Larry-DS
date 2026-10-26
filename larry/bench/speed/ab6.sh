#!/bin/bash
# ab6.sh - after run_all2.sh (the speed table, started by ab5.sh): whole prompts of 6K / 10K / 16K at depth 0 and a
# 3.5K turn at depth 32K, with the prefetch as the service has it (d0) and none at all with the CPU pool's light experts
# in every call up to 16K columns (dn) - where does the prefetch still pay?  Then tsvc2 on the restored service, AIME.
cd /opt/ai/dsv41
while pgrep -f "[r]un_all2.sh" > /dev/null; do sleep 10; done
echo "speed table done $(date +%T)"
TAB_SPECS="6000:0 10000:0 16000:0 3500:32768" TAB_REPS=4 bash bench/speed/tab_all.sh d0 \
  dn:STRATA_XE_PREFETCH_MIN=1000000,STRATA_XE_PREFETCH_MIN_B=1000000,STRATA_XE_CPU_LIGHT_MAX_N=16384
echo "A/B done $(date +%T)"
if curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health | grep -q 200; then
  echo "tsvc2 $(date +%T)"
  timeout 1200 bash tsvc2.sh 2>&1 | tail -16 | cut -c1-200
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
