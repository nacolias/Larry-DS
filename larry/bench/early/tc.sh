#!/bin/bash
# tl.sh - service off; the decode layer's device timeline (STRATA_GPU_TL) eager vs graph replay; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
run() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file p1.i32 -n 300
  echo "== $N ($*)"; grep -h "gpu timeline\|DSpark: [0-9]\|pass_v\|^decode:\|rc=\|abort\|rror" logs/$N.err | tail -5; }
pl() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160
  echo "== $N ($*)"; grep -h "^decode:\|rc=\|abort\|rror" logs/$N.err | tail -2; }
pl c0 STRATA_XE_CPU_ASYNC=0
pl c1 STRATA_XE_CPU_ASYNC=1
cmp -s logs/c0.out logs/c1.out && echo "r1 tokens == x0" || echo "r1 DIFFER: $(cmp logs/c0.out logs/c1.out | head -1)"
run dc0 STRATA_XE_CPU_ASYNC=0
run dc1 STRATA_XE_CPU_ASYNC=1

rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
