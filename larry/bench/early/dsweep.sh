#!/bin/bash
# dsweep.sh - DSpark depth / confidence sweep on the live service (defaults patched in ds41-xe-engine.sh, restored after)
cd /opt/ai/dsv41
cp ds41-xe-engine.sh /tmp/ds41-xe-engine.sh.orig
restart() {
  touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5; rm -f /run/lock/larry-profile-a.hold
  timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
  for i in $(seq 1 80); do tail -3 logs/engine.log | grep -q "serve: ready" && return 0; sleep 5; done; echo "not ready"; return 1
}
for cfg in "4 0.6" "5 0.6" "5 0.4" "4 0.4"; do
  set -- $cfg
  sed -e "s/STRATA_DSPARK_DEPTH:-4/STRATA_DSPARK_DEPTH:-$1/; s/STRATA_DSPARK_CONF:-0.6/STRATA_DSPARK_CONF:-$2/" /tmp/ds41-xe-engine.sh.orig > ds41-xe-engine.sh
  restart || continue
  echo "== depth $1 conf $2"
  timeout 1200 venv/bin/python svcbench.py 300 | tail -5
done
cp /tmp/ds41-xe-engine.sh.orig ds41-xe-engine.sh
restart
echo "restored: $(grep -c 'DEPTH:-4' ds41-xe-engine.sh)"
