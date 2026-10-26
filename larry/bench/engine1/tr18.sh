#!/bin/bash
# tr18.sh - engine1's draft: the promotions' copy chunks beside the draft's cross-card copies (math: its blocks and
# Markov chain ~50 % slower than code's).  Math at the service's settings, TP draft + stamps, 1000 tokens, clocks
# pinned: 1 MB chunks x 256 a pass (the same bytes, finer), 0.5 MB x 512, and one fill helper (STRATA_TP_PC_THREADS=1).
# No build.  Service held.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR18_CASES:-c1 c05 t1}; do
  case $c in
    c1) X="STRATA_XE_PROMOTE_CHUNK_MB=1 STRATA_TP_PROMOTE_CHUNKS=256" ;;
    c05) X="STRATA_XE_PROMOTE_CHUNK_MB=0.5 STRATA_TP_PROMOTE_CHUNKS=512" ;;
    t1) X="STRATA_TP_PC_THREADS=1" ;;
  esac
  N=tr18_math_$c
  env $B $D $T $REP STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_STAMP=1 $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file ${TR18_F:-pmath.i32} -n 1000 $A > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "promotion repacks" | head -1 | cut -c1-200
  tr "\r" "\n" < logs/$N.err | grep -a "tp draft stamps card 0" | tail -1
done
python3 -I bench/engine1/cyc.py logs/tr13_math.cpu logs/tr17_math.cpu logs/tr18_math_c1.cpu logs/tr18_math_c05.cpu logs/tr18_math_t1.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR18 done $(date +%T)"
