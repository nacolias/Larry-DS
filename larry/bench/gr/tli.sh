#!/bin/bash
# tli.sh - LONG-CONTEXT decode: the lightning indexer of decode / verify passes (batches below GGML_SYCL_LI_GEMM_MIN,
# default 32) runs the per-row kernel (~0.35 TFLOPS: at 98K context ~2.3 ms an indexer layer PER QUERY, 8 indexer
# layers, 2-5 queries a verify pass - ~45-60 ms of a ~96 ms pass); GGML_SYCL_LI_GEMM_MIN=1 sends every batch to the
# XMX GEMM path the prompts use (keys / queries are FP4 grid points in BF16: exact in FP16).  The user's service log
# (2026-10-22 00:00-00:23, agentic session): 24-31 tok/s at 90-100K context vs ~42-49 at <= 36K.
# 1) ~95K-token prompt (p30k.i32 three times), DSpark, 200 tokens; 32 / 1 alternating.  2) short context.  3) PPL.
cd /opt/ai/dsv41
source xe-env.sh
[ -f p95k.i32 ] || cat p30k.i32 p30k.i32 p30k.i32 > p95k.i32
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
for m in 32 1 32 1; do
  N=li$m$RANDOM
  env $B $D GGML_SYCL_LI_GEMM_MIN=$m RUN_TIMEOUT=1800 ./run.sh $N --ids-file p95k.i32 -n 200 --ctx 131072 --prompt-batch 16384 > /dev/null 2>&1
  echo "== LI_GEMM_MIN=$m: $(grep -h "^prompt:" logs/$N.err | cut -c1-70) | $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -v "^#" logs/$N.out | head -c 240 | tr "\n" " "; echo
done
# 2) short context (the essay prompt, DSpark, 400 tokens): no regression where the per-row kernel was fine
for m in 32 1 32 1; do
  N=lis$m$RANDOM
  env $B $D GGML_SYCL_LI_GEMM_MIN=$m RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== short, LI_GEMM_MIN=$m: $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ")"
done
# 3) decode-path PPL (2K tokens a step each): the same model within the noise (+-0.007)
for m in 32 1; do
  N=lip$m
  env $B GGML_SYCL_LI_GEMM_MIN=$m RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
  echo "[decode-path ppl, LI_GEMM_MIN=$m] $(grep -h PPL logs/$N.out)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
