#!/bin/bash
# tr19.sh - engine1's draft: off the copy engine (the promotions' copies queue there).  Rebuild libds41_tp (the draft's
# inputs / outputs and the passes' inputs by zero-copy kernels, STRATA_TP_ZC_IN; the Markov sums 16 bytes); then math
# and essay with the TP draft + stamps: zero-copy, and zero-copy + STRATA_TP_PUSH=1 (the cross-card sums by store
# kernels).  1000 tokens, the service's settings, clocks pinned.  Service held.
cd /opt/ai/dsv41/bench/engine1
TARGETS="${TR19_TARGETS:-ds41_tp}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR19_CASES:-math_zc math_push essay_zc essay_push}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in *_push) X="STRATA_TP_PUSH=1" ;; *) X="" ;; esac
  N=tr19_$c
  env $B $D $T $REP STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_STAMP=1 $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 $A > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "cross-card sums by\|tp prof" | tail -2 | cut -c1-230
  tr "\r" "\n" < logs/$N.err | grep -a "tp draft stamps card 0" | tail -1
done
python3 -I bench/engine1/cyc.py logs/tr13_math.cpu logs/tr17_math.cpu logs/tr19_math_zc.cpu logs/tr19_math_push.cpu
python3 -I bench/engine1/cyc.py logs/tr13_essay.cpu logs/tr17_essay.cpu logs/tr19_essay_zc.cpu logs/tr19_essay_push.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR19 done $(date +%T)"
