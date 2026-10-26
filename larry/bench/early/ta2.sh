#!/bin/bash
# ta.sh - service off; old build vs async dense (build-xs), plain decode and DSpark; multi-column R8 bench; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
OLD=/opt/ai/dsv41/xolotl/build-xe; NEW=/opt/ai/dsv41/xolotl/build-xs
run() { N=$1; shift; env "$@" STRATA_DEC_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 160
  echo "== $N ($*)"; grep -h "dec prof\|^decode:\|DSpark: [0-9]\|pass_v\|rc=\|abort\|rror" logs/$N.err | tail -5; }
run a0 STRATA_DENSE_ASYNC=0
run a1 STRATA_DENSE_ASYNC=1
run a2 STRATA_DENSE_ASYNC=1 GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1
grep -h "larry graph cache" logs/a2.err | tail -1
for n in a1 a2; do cmp -s logs/a0.out logs/$n.out && echo "$n tokens == a0" || echo "$n DIFFER: $(cmp logs/a0.out logs/$n.out | head -1)"; done
dsp() { N=$1; shift; env "$@" STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file p1.i32 -n 400
  echo "== $N ($*)"; grep -h "^decode:\|DSpark: [0-9]\|pass_v\|larry graph\|rc=\|abort\|rror" logs/$N.err | tail -5; }
dsp d1 STRATA_DENSE_ASYNC=1
dsp d2 STRATA_DENSE_ASYNC=1 GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
