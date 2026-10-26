#!/bin/bash
# ttp10.sh - engine1 at the exact service config (ctx 262144, 16K batches, FLEX 1 / margin 1.5, RESERVE 3.5): long and
# mid prompts with TP (the lazy KV replicas grow after the prompt; the prompt path's buffers on the expert cards) and
# today's path at 95K for comparison; DSpark, clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
[ -f p20k.i32 ] || head -c 80000 p95ke.i32 > p20k.i32
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; IDS=$2; NT=$3; shift 3; env $B "$@" RUN_TIMEOUT=1800 ./run.sh $N --ids-file $IDS -n $NT --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -ah "^prompt:" logs/$N.err | cut -c9-60) | $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "KV replicas now\|flex region .* -> \|promotion cache: [0-9]* promoted" | tail -4
  tr "\r" "\n" < logs/$N.err | grep -a "^tp prof" | tail -1 | cut -c1-260
  tr "\r" "\n" < logs/$N.err | grep -a "refused\|OUT_OF\|failed\|differs\|no room\|tp: no VRAM" | sort | uniq -c | head -4; }
run t10_ltp p95ke.i32 200 $D $T
run t10_mtp p20k.i32 200 $D $T
run t10_loff p95ke.i32 200 $D
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP10 done $(date +%T)"
