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
grep -h "larry graph cache" logs/a2.err | tail -2
run d0 STRATA_DENSE_ASYNC=0 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0
run d1 STRATA_DENSE_ASYNC=1 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0
run d2 STRATA_DENSE_ASYNC=1 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1
grep -h "larry graph cache" logs/d2.err | tail -2
run d3 STRATA_DENSE_ASYNC=1 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 GGML_SYCL_FASTMV_R8M_R=2
for n in a1 a2; do cmp -s logs/a0.out logs/$n.out && echo "$n tokens == a0" || echo "$n tokens DIFFER from a0 ($(cmp logs/a0.out logs/$n.out | head -1))"; done
export ONEAPI_DEVICE_SELECTOR=level_zero:3
for r in 0 2 4; do echo "== r8m R=$r"; (cd bench/fm2 && GGML_SYCL_FASTMV_R8M_R=$r ./fastmv_bench_r8m 300 | grep "R8.*ncols 5"); done
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
