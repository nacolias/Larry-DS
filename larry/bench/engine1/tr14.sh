#!/bin/bash
# tr14.sh - engine1's DSpark draft on the TP cards, step 1: build (the planar kernels' Q2_K form, the TP core's draft,
# the engine's STRATA_TP_DRAFT), the Q2_K parity test (cards 1 / 2), then one-shot runs at the service's settings with
# the replicas: the check mode (both drafts side by side: agreement by slot, 400 tokens) and - when it passed - the
# timing runs (1000 tokens: essay / code / math with STRATA_TP_DRAFT=1, clocks pinned; cyc.py against tr13's).
# Service held.
cd /opt/ai/dsv41/bench/engine1
t0=$(date +%s)
# the build first: build-xe.sh sources oneAPI itself (a second source returns 3 under its set -e)
TARGETS="${TR14_TARGETS:-ds41_xe ds41_tp tp_bench test_xe_q2k ds41_generate_xe}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
echo "build $(( $(date +%s) - t0 )) s"
cd /opt/ai/dsv41
source xe-env.sh
if [ -z "$TR14_NOPARITY" ]; then
  for j in 0 1; do
    echo "== test_xe_q2k JOBNT=$j (block 0, card 1)"
    STRATA_XE_JOBNT=$j timeout 300 xolotl/build-xe/test_xe_q2k $SIDECAR --block 0 --experts 0,17,77,127 --device 1 --bench 200 2>&1 | tail -21
  done
  echo "== test_xe_q2k JOBNT=1 (block 2, card 2)"
  STRATA_XE_JOBNT=1 timeout 300 xolotl/build-xe/test_xe_q2k $SIDECAR --block 2 --experts 3,64 --device 2 2>&1 | tail -10
fi
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
line() {
  echo "== $1 $(grep -ah "^decode:" logs/$1.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$1.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$1.err)"
  tr "\r" "\n" < logs/$1.err | grep -a "tp draft\|DSpark step\|gave up\|error" | grep -v "^tp draft check" | head -4 | cut -c1-240
  tr "\r" "\n" < logs/$1.err | grep -a "^tp draft check" | tail -2
  head -c 240 logs/$1.out | tr "\n" " "; echo
}
if [ -z "$TR14_NOCHECK" ]; then
  N=tr14_check
  env $B $D $T $REP STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_CHECK=1 RUN_TIMEOUT=1200 ./run.sh $N --ids-file pe.i32 -n 400 $A > /dev/null 2>&1
  line $N
  grep -q "^rc=0" logs/$N.err || { echo "CHECK RUN FAILED"; /usr/local/sbin/larry-gpu-clock restore > /dev/null; dmesg -T | grep -c "CAT error"; exit 1; }
fi
for c in ${TR14_CASES:-essay code math}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  N=tr14_$c
  env $B $D $T $REP STRATA_TP_DRAFT=1 STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 $A > /dev/null 2>&1
  line $N
done
python3 -I bench/engine1/cyc.py logs/tr13_essay.cpu logs/tr14_essay.cpu
python3 -I bench/engine1/cyc.py logs/tr13_code.cpu logs/tr14_code.cpu
python3 -I bench/engine1/cyc.py logs/tr13_math.cpu logs/tr14_math.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR14 done $(date +%T)"
