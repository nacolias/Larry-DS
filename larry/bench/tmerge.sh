#!/bin/bash
# tmerge.sh - STRATA_HC_MERGE (hc_fa + hc_fb as one kernel) on vs off, everything else of the fusion round on;
# plain decode x2 and DSpark x3 each, interleaved, clocks pinned, service env + bias 0.5 / 0.3.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)" | sed "s|STRATA_DSPARK=[^ ]* STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K|DSpark|"; }
for i in 1 2; do run mp_on$i; run mp_off$i STRATA_HC_MERGE=0; done
for i in 1 2 3; do run md_on$i $D; run md_off$i $D STRATA_HC_MERGE=0; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TMERGE done $(date +%T)"
