#!/bin/bash
# tut.sh - unitrace of plain decode with the attention output split (eager, every command direct so all kernels show):
# per-kernel device times on card 0 and the expert cards around the split
cd /opt/ai/dsv41
/usr/local/sbin/larry-gpu-clock pin > /dev/null
export STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=4.2 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0
export STRATA_XE_ARM=0 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=0 STRATA_TP_GRAPH=0
RUN_TIMEOUT=1200 ./utrace.sh tpu 6 --ids-file pe.i32 -n 200 > /dev/null 2>&1
ls -la logs/ut-tpu | head
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
