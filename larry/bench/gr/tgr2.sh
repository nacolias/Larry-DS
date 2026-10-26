#!/bin/bash
# tgr2.sh - with the dense card dispatching the experts (STRATA_XE_GROUTE=1) the host has slack after each readback:
# do the multi-card splits that were host-bound before pay now?  DSpark essay + plain decode, alternating:
#   base   GROUTE=1
#   tp     + STRATA_TP_OPROJ=2 (the attention output over four cards; needs the 4.2 GiB expert-card reserve)
#   hs30   + STRATA_HEAD_SPLIT_MB=30 (q_b / wo_b / the shared expert's gate-up split by rows over the cards: -12 %
#          before, when the host reached the tier 64 us after each readback)
#   ag     + STRATA_XE_ARM_GRAPH=1 (each expert card's armed share one replayed graph)
# STRATA_XE_JOBNT=${JOBNT:-0}.  The reserve is 4.2 for every variant (same expert placement).
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=4.2 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 STRATA_XE_JOBNT=${JOBNT:-0} STRATA_XE_GROUTE=1 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "STRATA_XE_GROUTE: [0-9]* routed\|not split\|split [0-9]* calls" logs/$N.err | sort | uniq -c | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"; }
for i in 1 2; do
  for v in "base" "tp STRATA_TP_OPROJ=2" "hs30 STRATA_HEAD_SPLIT_MB=30" "ag STRATA_XE_ARM_GRAPH=1"; do
    set -- $v; tag=$1; shift
    run gd_${tag}_$i $D "$@"
    run gp_${tag}_$i "$@"
  done
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
