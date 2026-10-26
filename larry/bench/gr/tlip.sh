#!/bin/bash
# tlip.sh - decode-path PPL repeats (2K tokens, a step each): GGML_SYCL_LI_GEMM_MIN 32 / 1 alternating x2
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256"
for m in 1 32 1 32; do
  N=lipr$m$RANDOM
  env $B GGML_SYCL_LI_GEMM_MIN=$m RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
  echo "[decode-path ppl, LI_GEMM_MIN=$m] $(grep -h PPL logs/$N.out)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
