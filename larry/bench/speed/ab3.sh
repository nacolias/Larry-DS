#!/bin/bash
# ab3.sh - the prefetch thresholds with STRATA_XE_CPU_LIGHT=4 (the service's): whole batches from 4096 tokens, an
# enc-skip batch's upper layers from 2048 (STRATA_XE_PREFETCH_MIN_B) - pp2048 / pp2900 at depth 0, pp512 at 32K (its
# first message: a 31K prompt, the regression to watch).  Waits for the AIME client (STOP file: it ends after its
# current problem), runs tab_all.sh (the service held, the configs, the service back), then starts AIME again.
cd /opt/ai/dsv41
while pgrep -f "[a]ime.py" > /dev/null; do sleep 10; done
echo "AIME client gone $(date +%T)"
TAB_SPECS="2048:0 2900:0 512:32768" TAB_REPS=5 bash bench/speed/tab_all.sh p0 \
  p1:STRATA_XE_PREFETCH_MIN=4096,STRATA_XE_PREFETCH_MIN_B=2048
echo "A/B done $(date +%T)"
if curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health | grep -q 200; then
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
