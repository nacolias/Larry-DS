#!/bin/bash
# tcpu2.sh - engine1's promotion pipeline A/B (the CPU-pool experts of TP verify passes): the DSpark essay at the
# service's settings, clocks pinned, the tier's CPU log on.  Cases (TCPU_CASES): base (today), pub (finished promotions
# published before each pass), early (+ bounces filled at the CPU use), early1 (+ promoted at the first use), early1c
# (+ 256 copy chunks of 4 MB between passes), chk (early1c with every published slot checked against a fresh repack)
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
E1="STRATA_TP_PC_EARLY=1 STRATA_TP_PC_PUBLISH=1"
C="STRATA_TP_PROMOTE_CHUNKS=256 STRATA_XE_PROMOTE_CHUNK_MB=4"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TCPU_CASES:-base pub early early1 early1c chk}; do
  case $c in
    base*) X="STRATA_TP_PC_EARLY=0 STRATA_TP_PC_PUBLISH=0" ;;
    pub*) X="STRATA_TP_PC_EARLY=0 STRATA_TP_PC_PUBLISH=1" ;;
    early1c*) X="$E1 STRATA_XE_PROMOTE_MIN=1 $C" ;;
    early1*) X="$E1 STRATA_XE_PROMOTE_MIN=1" ;;
    early*) X="$E1" ;;
    chk*) X="$E1 STRATA_XE_PROMOTE_MIN=1 $C STRATA_XE_PROMOTE_CHECK=1" ;;
  esac
  N=tcpu2_$c
  env $B $D $T $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file pe.i32 -n 400 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /\|cpu [0-9]* ms" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "tp prof" | tail -2 | cut -c1-230
  tr "\r" "\n" < logs/$N.err | grep -a "promotion cache: [0-9]* promoted\|promotion repacks" | tail -2 | cut -c1-200
  python3 -I bench/engine1/cpulog.py logs/$N.cpu 2>&1 | grep -v "layers with the most"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TCPU2 done $(date +%T)"
