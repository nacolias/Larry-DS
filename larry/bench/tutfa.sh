#!/bin/bash
# tutfa.sh - why STRATA_ATTN_FUSE gives no tok/s: (1) STRATA_DEC_PROF host phases per decode layer (attention graph
# submit, readback wait, ...), (2) unitrace of plain decode (ARM off for the tracer): card 0's per-kernel GPU times, for
# STRATA_ATTN_FUSE 0 vs 1.  Service env, exact routing.  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
RX="--route-bias 0 --route-bias-file 0"
for f in 0 1; do
  N=dp_fa$f
  env $B STRATA_DEC_PROF=1 STRATA_ATTN_FUSE=$f RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 200 $RX > /dev/null 2>&1
  echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "^dec prof" logs/$N.err | tail -1) $(grep -h "^rc=" logs/$N.err)"
done
for f in 0 1; do
  env $B STRATA_XE_ARM=0 STRATA_ATTN_FUSE=$f RUN_TIMEOUT=900 ./utrace.sh utfa$f 6 --ids-file pe.i32 -n 160 $RX > /dev/null 2>&1
  echo "== utfa$f $(grep -h "^decode:" logs/utfa$f.err | grep -o "= [0-9.]* tok/s") $(grep -h "^rc=" logs/utfa$f.err)"
  python3 utseq.py logs/ut-utfa$f "#0" 1 > logs/utfa$f.seq 2>&1
  grep -A 40 "layers: span" logs/utfa$f.seq | head -40
done
echo "TUTFA done $(date +%T)"
