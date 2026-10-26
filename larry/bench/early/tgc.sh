#!/bin/bash
# tgc.sh - plain greedy decode (no DSpark) with ggml-sycl's SYCL-graph cache off / on, GPU clocks pinned as the
# service does: tokens must match; decode tok/s and the dense host profile (STRATA_DEC_PROF) per run
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DEC_PROF=1"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file p1.i32 -n 160 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | cut -c1-48) | $(grep -h "dec prof" logs/$N.err | tail -1 | cut -c1-150)"
  grep -h "not recordable\|graph cache\|eager" logs/$N.err | sort | uniq -c | head -5 | sed "s/^/     /"; }
for i in 1 2; do
  run gc0$i GGML_SYCL_GRAPH_CACHE=0
  run gc1$i GGML_SYCL_GRAPH_CACHE=1 GGML_SYCL_GRAPH_STATS=1
done
for i in 1 2; do for j in 0 1; do cmp -s logs/gc0$i.out logs/gc01.out; a=$?; cmp -s logs/gc$j$i.out logs/gc01.out && echo "gc$j$i tokens == gc01" || echo "gc$j$i tokens DIFFER from gc01"; done; done
/usr/local/sbin/larry-gpu-clock restore
echo "CAT error times: $(dmesg -T | grep "CAT error" | awk '{print $4}' | sort -u | tr "\n" " ")"
