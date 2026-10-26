#!/bin/bash
# run_all.sh - the speed table: the service held and stopped, tspeed.sh with the service's settings (DSpark on) and
# without the draft (plain: STRATA_SERVE_DSPARK empty), then the service back (larry-mode ds41) - unless an engine is
# still running, then the hold stays and nothing is started.
set -u
cd /opt/ai/dsv41
S=bench/speed
cat_errs() { dmesg | grep -c -i "cat error"; }
echo "start $(date '+%F %T'), CAT errors in dmesg: $(cat_errs)"
touch /run/lock/larry-profile-a.hold
systemctl stop larry-ds41
for i in $(seq 1 60); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running after the stop: aborting, hold kept"; exit 1; fi
echo "service stopped $(date +%T)"
bash $S/tspeed.sh dspark
bash $S/tspeed.sh plain STRATA_SERVE_DSPARK=
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running: service NOT restarted, hold kept"; exit 1; fi
echo "CAT errors in dmesg: $(cat_errs)"
rm -f /run/lock/larry-profile-a.hold
larry-mode ds41
c=0
for i in $(seq 1 60); do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health)
  [ "$c" = "200" ] && break
  sleep 5
done
echo "service health $c $(date '+%F %T')"
