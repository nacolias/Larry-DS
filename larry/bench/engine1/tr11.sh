#!/bin/bash
# tr11.sh - engine1 replicas, step 4: the merged usage profile (/opt/ai/dsv41/replica-usage.txt: essay + code + math),
# 2.5 GiB at the 1.0 GiB margin (the new default).  Math without / with, essay and code with; then the service's test
# sequence twice on a private port with the replicas on (prompts release them, the decode after re-uploads them).
cd /opt/ai/dsv41/bench/engine1
TARGETS="ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
(
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR11_CASES:-math_r0 math_um essay_um code_um}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in *_r0*) X="STRATA_XE_REPLICA_GIB=0" ;; *_um*) X="$REP" ;; esac
  N=tr11_$c
  env $B $D $T $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "replicas: [0-9]* on" | cut -c1-200
done
python3 -I bench/engine1/cyc.py logs/tr6_math.cpu logs/tr11_math_r0.cpu logs/tr11_math_um.cpu
python3 -I bench/engine1/cyc.py logs/tr9_essay_r0.cpu logs/tr10_essay_r0b.cpu logs/tr10_essay_ue.cpu logs/tr10_essay_ue2.cpu logs/tr11_essay_um.cpu
python3 -I bench/engine1/cyc.py logs/tr9_code_r0.cpu logs/tr9_code_r2.cpu logs/tr11_code_um.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
)
TPG_ROUNDS=2 bench/engine1/tpg.sh rep $REP 2>&1
grep -h "replicas:" logs/engine-rep.log | sort | uniq -c | sort -rn | head -5
echo "TR11 done $(date +%T)"
