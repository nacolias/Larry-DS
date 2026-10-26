#!/bin/bash
# tpg_light.sh - the deployment check for STRATA_XE_CPU_LIGHT=4: the service held and stopped, bench/engine1/tpg.sh
# with it (the tsvc2 sequence on 18081: needles 200 / 8K x2 / 16K / 31K, svcbench, essay, the tool call), then the
# service back on its unchanged config.  Deploying is a separate step.
cd /opt/ai/dsv41
cat_errs() { dmesg | grep -c -i "cat error"; }
echo "start $(date '+%F %T'), CAT errors: $(cat_errs)"
touch /run/lock/larry-profile-a.hold
systemctl stop larry-ds41
for i in $(seq 1 60); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running after the stop: aborting, hold kept"; exit 1; fi
bash bench/engine1/tpg.sh cl4 STRATA_XE_CPU_LIGHT=4
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running: service NOT restarted, hold kept"; exit 1; fi
echo "CAT errors: $(cat_errs); segfaults today: $(dmesg | grep -c segfault)"
rm -f /run/lock/larry-profile-a.hold
larry-mode ds41 > /dev/null 2>&1
c=0
for i in $(seq 1 60); do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health)
  [ "$c" = "200" ] && break
  sleep 5
done
echo "service health $c $(date '+%F %T')"
