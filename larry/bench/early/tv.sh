#!/bin/bash
# tv.sh - service off; fastmv column scaling; DSpark one-shot with more VRAM experts (card 0 share, card reserve); on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
source xe-env.sh
for r in 0 2; do echo "== fastmv R8 by columns, R8M_R=$r"; (cd bench/fm2 && ONEAPI_DEVICE_SELECTOR=level_zero:3 GGML_SYCL_FASTMV_R8M_R=$r ./fastmv_bench_nc 200 | grep "q_b R8\|o_b R8"); done
mon() { while true; do echo "$(date +%T) $(xpu-smi stats -d 0 2>/dev/null | grep -m1 'GPU Memory Used' | awk '{print $NF}')"; sleep 2; done; }
run() { N=$1; shift; (mon > logs/$N.vram &) ; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0 ./run.sh $N --ids-file p1.i32 -n 300
  pkill -f "xpu-smi stats -d 0" ; kill %1 2>/dev/null
  echo "== $N ($*)"; grep -h "seeded with\|card [0-3]:\|DSpark: [0-9]\|pass_v\|^decode:\|rc=\|abort\|rror" logs/$N.err | tail -9; echo "card 0 max MiB: $(awk '{print $2}' logs/$N.vram | sort -n | tail -1)"; }
run v0
run v1 STRATA_XE_DENSE_GIB=13 STRATA_XE_RESERVE_GIB=2.5
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
systemctl is-active larry-ds41
