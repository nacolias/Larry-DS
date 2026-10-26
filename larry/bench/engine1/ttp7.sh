#!/bin/bash
# ttp7.sh - engine1 A/B at the SERVICE's settings (ds41-xe-engine.sh): ctx 262144, 16K prompt batches, FLEX 1 GiB /
# margin 1.5; DSpark x3 + plain x2 each, interleaved (essay x400, bias 0.5 / 0.3); then a 300-token prompt on each
# (the short-prompt path that ran out of VRAM in the FLEX 3 service attempt); clocks pinned, service held down
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=${FLEX:-1} STRATA_XE_FLEX_MARGIN_GIB=${FM:-1.5}"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
[ -f p300.i32 ] || head -c 1200 p95ke.i32 > p300.i32
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; IDS=$2; NT=$3; shift 3; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file $IDS -n $NT --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^prompt:" logs/$N.err | cut -c9-48) | $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "refused\|OUT_OF\|failed\|no room" logs/$N.err | sort | uniq -c | head -4; }
for i in 1 2 3; do run sv_doff$i pe.i32 400 $D; run sv_dtp$i pe.i32 400 $D $T; done
for i in 1 2; do run sv_poff$i pe.i32 400; run sv_ptp$i pe.i32 400 $T; done
run sv_soff p300.i32 64 $D; run sv_stp p300.i32 64 $D $T
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP7 done $(date +%T)"
