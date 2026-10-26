#!/bin/bash
# run3.sh - STRATA_QKV_FUSE (decode / DSpark / 31.6K prompt / both PPLs), STRATA_CONT_SKIP (decode), then tmv.sh.
# Baseline env: the long-context wins (LI_GEMM_MIN=1, LI_XMX=1) + JOBNT=1 + COMBINE=1, GROUTE off.
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N $ARGS > /dev/null 2>&1
  echo "== $N ($*) $(grep -h "^prompt:" logs/$N.err | cut -c1-48) | $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "q_a+kv pairs" logs/$N.err | head -1) $(grep -h "^rc=" logs/$N.err)"; }
ARGS="--ids-file pe.i32 -n 400"
for i in 1 2; do
  run qp0_$i STRATA_QKV_FUSE=0; run qp1_$i STRATA_QKV_FUSE=1
  run qd0_$i $D STRATA_QKV_FUSE=0; run qd1_$i $D STRATA_QKV_FUSE=1
done
ARGS="--ids-file p30k.i32 -n 8 --prompt-batch 16384"
run qr0 STRATA_QKV_FUSE=0; run qr1 STRATA_QKV_FUSE=1
for q in 0 1; do
  N=qpp$q; env $B STRATA_QKV_FUSE=$q RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 2048 > /dev/null 2>&1
  echo "[prompt-path ppl, QKV_FUSE=$q] $(grep -h PPL logs/$N.out)"
  N=qdp$q; env $B STRATA_QKV_FUSE=$q RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
  echo "[decode-path ppl, QKV_FUSE=$q] $(grep -h PPL logs/$N.out)"
done
ARGS="--ids-file pe.i32 -n 400"
for i in 1 2; do run cs0_$i STRATA_CONT_SKIP=0; run cs1_$i STRATA_CONT_SKIP=1; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "== tmv $(date +%T)"; bench/gr/tmv.sh > bench/gr/tmv.log 2>&1; echo "tmv rc=$?"
echo "ALL3 done $(date +%T)"
