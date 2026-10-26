#!/bin/bash
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
pl() { N=$1; shift; env "$@" STRATA_DEC_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160; echo "== $N"; grep -h "decode step\|^decode:\|rc=\|rror\|abort" logs/$N.err | tail -3; }
pl u1
pl u2
dsp() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file p1.i32 -n 300; echo "== $N"; grep -h "DSpark: [0-9]\|pass_v\|^decode:\|rc=\|rror\|abort" logs/$N.err | tail -4; }
dsp du1
tr "\n" " " < logs/u1.out | python3 tok.py dec - 2>&1 | head -c 400; echo
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
