#!/bin/bash
# ut12.sh - service off; unitrace timelines of plain decode (ut1) and DSpark decode (ut2); service back on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
STRATA_DENSE_REQUANT=q6_K ./utrace.sh ut1 1.5 --ids-file p1.i32 -n 128
STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 STRATA_DENSE_REQUANT=q6_K ./utrace.sh ut2 1.5 --ids-file p1.i32 -n 160
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
for n in ut1 ut2; do echo "== $n"; grep -h "^decode\|DSpark: [0-9]\|pass_v\|rc=" logs/$n.err | tail -4; ls -la logs/ut-$n | tail -3; done
systemctl is-active larry-ds41
