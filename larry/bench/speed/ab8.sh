#!/bin/bash
# ab8.sh - DSpark's long prompts from empty (6.7K / 31K / 97K: pp512's first messages at depths 8192 / 32768 / 98304,
# the speed table's order) with STRATA_XE_CPU_LIGHT 0 (z0), 4 as built (z4), then 4 with patch-cpulight2.py (the pool's
# sum read from pinned host memory: no buffer on card 0; z4b).  The service is held and stopped already (after the GT
# resets of 10:41); at the end it comes back, tsvc2 runs, AIME starts again.
cd /opt/ai/dsv41
cat_errs() { dmesg | grep -c -i -E "cat error|engine reset"; }
echo "start $(date +%T), CAT errors / engine resets: $(cat_errs)"
export TAB_SPECS="512:8192,32768,98304" TAB_REPS=3
bash bench/speed/tab.sh z0 STRATA_XE_CPU_LIGHT=0
bash bench/speed/tab.sh z4
python3 -I patch-cpulight2.py && (TARGETS=ds41_xe nice -n 10 bash build-xe.sh > logs/build-cpulight2.log 2>&1; echo "build rc=$?")
bash bench/speed/tab.sh z4b
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running: service NOT restarted, hold kept"; exit 1; fi
echo "CAT errors / engine resets: $(cat_errs)"
rm -f /run/lock/larry-profile-a.hold
larry-mode ds41 > /dev/null 2>&1
c=0
for i in $(seq 1 60); do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health)
  [ "$c" = "200" ] && break
  sleep 5
done
echo "service health $c $(date +%T)"
if [ "$c" = "200" ]; then
  echo "tsvc2 $(date +%T)"
  timeout 1200 bash tsvc2.sh 2>&1 | tail -16 | cut -c1-200
  rm -f bench/quality/out/STOP
  cd bench/quality && (setsid nohup taskset -c 15 /opt/ai/dsv41/venv/bin/python -I aime.py --deadline 14:30 >> out/aime-1.log 2>&1 < /dev/null &)
  echo "AIME restarted $(date +%T)"
fi
