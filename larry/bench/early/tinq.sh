#!/bin/bash
# tinq.sh - plain decode (no DSpark) with the in-kernel quantisation on / off, essay prompt x2 each, clocks pinned;
# the GPU timeline for the per-layer split
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"; }
for i in 1 2; do
  run iq1$i GGML_SYCL_LARRY_INQ=1
  run iq0$i GGML_SYCL_LARRY_INQ=0
done
run iqt1 GGML_SYCL_LARRY_INQ=1 STRATA_GPU_TL=1
grep -h "gpu timeline" logs/iqt1.err | tail -1 | grep -o "0->1 [0-9.]*\|1->2 [0-9.]*\|2->15 [0-9.]*\|15->16 [0-9.]*\|16->3 [0-9.]*" | tr "\n" " "; echo
run iqt0 GGML_SYCL_LARRY_INQ=0 STRATA_GPU_TL=1
grep -h "gpu timeline" logs/iqt0.err | tail -1 | grep -o "0->1 [0-9.]*\|1->2 [0-9.]*\|2->15 [0-9.]*\|15->16 [0-9.]*\|16->3 [0-9.]*" | tr "\n" " "; echo
/usr/local/sbin/larry-gpu-clock restore > /dev/null
