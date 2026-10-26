#!/bin/bash
# tmmid.sh - the DSpark draft's Q2_K experts: device-side mul_mat_id (GGML_SYCL_LARRY_MMID_Q2K=1) vs the routed path,
# both with the draft's dense in Q6_K, clocks pinned; essay prompt x2 each
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.6 STRATA_DSPARK_PROF=1 STRATA_DSPARK_DENSE_TYPE=q6_K STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-46)"
  grep -h "DSpark: [0-9]* verify\|dspark prof" logs/$N.err | tail -2 | sed "s/^/     /"; }
for i in 1 2; do
  run mm1$i GGML_SYCL_LARRY_MMID_Q2K=1
  run mm0$i GGML_SYCL_LARRY_MMID_Q2K=0
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
