#!/bin/bash
# tr6.sh - engine1 round 7: rebuild (exact XMX mat-vecs: B = q - 32, the scale after the MAD; the tier's per-expert use
# counter STRATA_XE_USAGE_OUT), then (a) tp_bench n 5 base vs the new options (and the tuner's report), (b) the 5-token
# decode-path PPL with the tuned forms twice (vs tppl5's p5_base1 / p5_base2), (c) usage profiles of three texts
# (essay, code, math: 1000 tokens each, the new options, every layer's stamps) for the placement study.  Service held.
cd /opt/ai/dsv41/bench/engine1
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
BT=/opt/ai/dsv41/xolotl/build-xe/tp_bench
M=/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf
NEW="STRATA_TP_ROUTE_PAR=1 STRATA_TP_NJ_DEV=1 STRATA_TP_JOB_GRID=8 STRATA_TP_MV_TUNE=1"
mkdir -p bench/engine1/prof9
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for o in base new; do
  X="STRATA_TP_ROUTE_PAR=1"; [ $o = new ] && X="$NEW"
  env STRATA_XE_JOBNT=1 $X $BT -m $M -N 4 --pos 1000 -n 5 --iters 40 --graph 1 --cache /srv/dsv41/tpcache --experts 48 > bench/engine1/prof9/g_${o}.out 2>&1
  echo "== bench n 5 $o: $(grep -h "pass median" bench/engine1/prof9/g_${o}.out | grep -o 'median [0-9.]* us')"
done
grep -h "us a launch" bench/engine1/prof9/g_new.out | cut -c1-260
/usr/local/sbin/larry-gpu-clock restore > /dev/null
CASES="tune3 tune4" bench/engine1/tppl5.sh 2>&1 | grep -v "us a launch"
venv/bin/python -I bench/engine1/klrows.py p5_base1 p5_base2 p5_tune3 p5_tune4 2>&1
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for p in essay code math; do
  case $p in essay) F=pe.i32 ;; code) F=pcode.i32 ;; math) F=pmath.i32 ;; esac
  N=tr6_$p
  env $B $D $T $NEW STRATA_TP_STAMP_LAYER=-1 STRATA_XE_USAGE_OUT=logs/use_$p.txt STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err) | use lines $(grep -c '^U' logs/use_$p.txt)"
  tr "\r" "\n" < logs/$N.err | grep -a "tp stamps all layers" | tail -1
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TR6 done $(date +%T)"
