#!/bin/bash
# ttp12.sh - engine1 vs today's path across context lengths at the service config (ctx 262144, 16K batches, FLEX 1 /
# margin 1.5, RESERVE 3.5), both with the tier's slab fill (STRATA_XE_FILL_SLAB): DSpark on the essay (x2 each), the
# 20K and 95K prefixes (x1 each), plain decode (x1 each); bias 0.5 / 0.3, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; IDS=$2; NT=$3; shift 3; env $B "$@" RUN_TIMEOUT=1800 ./run.sh $N --ids-file $IDS -n $NT --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -ah "^prompt:" logs/$N.err | cut -c9-60) | $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "more experts in the cards' last slabs\|card [1-3]:.*VRAM free\|flex region .* -> \|LM head split over\|no room" | cut -c1-160 | sort | uniq -c | head -8
  tr "\r" "\n" < logs/$N.err | grep -a "refused\|OUT_OF\|failed\|differs\|tp: no VRAM" | sort | uniq -c | head -4; }
for i in 1 2; do run t12_eoff$i pe.i32 400 $D; run t12_etp$i pe.i32 400 $D $T; done
run t12_moff p20k.i32 200 $D; run t12_mtp p20k.i32 200 $D $T
run t12_loff p95ke.i32 200 $D; run t12_ltp p95ke.i32 200 $D $T
run t12_poff pe.i32 400; run t12_ptp pe.i32 400 $T
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP12 done $(date +%T)"
