#!/bin/bash
# tgr.sh - STRATA_XE_GROUTE (the dense card dispatches decode / verify calls): 1) correctness - the route kernel's lists
# against the host's grouping on every call (STRATA_XE_GROUTE_CHECK=1) in decode-path PPL (512 tokens, a step each)
# and a DSpark run (verify passes); 2) speed - plain decode and DSpark on the essay prompt, off / on alternating, with
# STRATA_XE_JOBNT=${JOBNT:-0}; 3) one timeline run each (STRATA_GPU_TL: the expert cards' readback -> start lag).
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 STRATA_XE_JOBNT=${JOBNT:-0} GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
gr() { grep -h "STRATA_XE_GROUTE: [0-9]* routed\|GROUTE_CHECK\|armed card shares" logs/$1.err | tr "\n" " "; }
# 1) correctness (p512.i32: the first 512 tokens of p2k.i32)
[ -f p512.i32 ] || head -c 2048 p2k.i32 > p512.i32
N=grck_ppl
env $B STRATA_XE_GROUTE=1 STRATA_XE_GROUTE_CHECK=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file p512.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
echo "[check, decode-path ppl] $(grep -h PPL logs/$N.out) | $(gr $N) $(grep -h "^rc=" logs/$N.err)"
N=grck_ds
env $B $D STRATA_XE_GROUTE=1 STRATA_XE_GROUTE_CHECK=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 120 > /dev/null 2>&1
echo "[check, DSpark] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(gr $N) $(grep -h "^rc=" logs/$N.err)"
grep -v "^#" logs/$N.out | head -c 300 | tr "\n" " "; echo
# 2) speed
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(gr $N)"; }
for i in 1 2; do
  run pd0_$i STRATA_XE_GROUTE=0
  run pd1_$i STRATA_XE_GROUTE=1
  run ds0_$i $D STRATA_XE_GROUTE=0
  run ds1_$i $D STRATA_XE_GROUTE=1
done
# 3) timelines
for g in 0 1; do
  N=grtl$g
  env $B $D STRATA_XE_GROUTE=$g STRATA_GPU_TL=1 STRATA_XE_PROF=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 300 > /dev/null 2>&1
  echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s")"
  grep -h "gpu timeline" logs/$N.err | tail -1 | tr ";" "\n" | grep -v promotion | head -30
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
