#!/bin/bash
# tr17.sh - engine1's draft, step 4: promotions without kernels (the helper streams the planes into the bounce:
# STRATA_TP_PC_DEVREPACK default 0, the 32-block host repack; chunks on the copy engine), the draft's Q2_K experts in
# kernels of their own (the trunk's expert kernels as they were).  Build; test_xe_q2k; tp_bench (n 5 / 1: back to the
# kernel round's 22.06 / ~13.3 ms?); the essay with STRATA_XE_PROMOTE_CHECK=1 (every promoted slot against a fresh
# repack); then one-shot DSpark timing at the service's settings with the replicas, 1000 tokens, clocks pinned: essay /
# code / math with STRATA_TP_DRAFT=1 (+ stamps), the essay with the ggml draft.  Service held.
cd /opt/ai/dsv41/bench/engine1
t0=$(date +%s)
TARGETS="${TR17_TARGETS:-ds41_xe ds41_tp tp_bench test_xe_q2k ds41_generate_xe}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
echo "build $(( $(date +%s) - t0 )) s"
cd /opt/ai/dsv41
source xe-env.sh
echo "== test_xe_q2k"
STRATA_XE_JOBNT=1 timeout 300 xolotl/build-xe/test_xe_q2k $SIDECAR --block 1 --experts 5,99 --device 1 --bench 100 2>&1 | tail -16
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for n in 5 1; do
  echo "== tp_bench n $n: $(STRATA_XE_JOBNT=1 timeout 600 xolotl/build-xe/tp_bench -m $GGUF -N 4 --pos 1000 -n $n --iters 40 --graph 1 --cache /srv/dsv41/tpcache --experts 48 2>&1 | grep -o 'median [0-9.]* us')"
done
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
for c in ${TR17_CASES:-essay_chk essay code math essay_ggml}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in *_ggml) X="STRATA_TP_DRAFT=0" ;; *_chk) X="STRATA_TP_DRAFT=1 STRATA_XE_PROMOTE_CHECK=1" ;; *) X="STRATA_TP_DRAFT=1 STRATA_TP_DRAFT_STAMP=1" ;; esac
  N=tr17_$c
  env $B $D $T $REP $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n ${TR17_N:-1000} $A > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "seeded:\|promotion repacks\|slot differs\|promotions:\|bad slot" | head -4 | cut -c1-200
  tr "\r" "\n" < logs/$N.err | grep -a "tp draft stamps" | tail -4
done
python3 -I bench/engine1/cyc.py logs/tr13_essay.cpu logs/tr17_essay_ggml.cpu logs/tr17_essay.cpu
python3 -I bench/engine1/cyc.py logs/tr13_code.cpu logs/tr17_code.cpu
python3 -I bench/engine1/cyc.py logs/tr13_math.cpu logs/tr17_math.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR17 done $(date +%T)"
