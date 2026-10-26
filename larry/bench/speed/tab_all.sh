#!/bin/bash
# tab_all.sh NAME:VAR=v,VAR=v ... - the short-prompt A/B: the service held and stopped, tab.sh per config (in order),
# then the service back (larry-mode ds41) - unless an engine is still running, then the hold stays and nothing
# is started.  Example: tab_all.sh l0:STRATA_XE_CPU_LIGHT=0 l4:STRATA_XE_CPU_LIGHT=4
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
for CFG in "$@"; do
  N=${CFG%%:*}
  E=${CFG#*:}
  [ "$E" = "$CFG" ] && E=
  bash $S/tab.sh $N ${E//,/ }
done
[ -n "${TAB_PPL:-}" ] && bash $S/tlppl.sh
if [ -n "${TAB_ONESHOT:-}" ]; then   # the tier's close() after a CPU-light prompt (2026-10-24: it crashed there once)
  STRATA_XE_CPU_LIGHT=4 STRATA_DENSE_REQUANT=q6_K STRATA_ENGRAM_REQUANT=q6_K RUN_TIMEOUT=600 ./run.sh lp_close --ids-file p512.i32 -n 8
  echo "one-shot close check: $(grep -h "^prompt:" logs/lp_close.err | cut -c1-100) $(grep -h "^rc=" logs/lp_close.err)"
fi
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
