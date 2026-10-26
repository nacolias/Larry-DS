#!/bin/bash
# tlppl-s1k.sh - numerics of 1024-column dense spans (--prefill-chunk 1024 + GGML_SYCL_DSV4_SDPA_N=1024): one-shot
# prompt-path runs with the service's dense settings of p2k at --prompt-batch 2048 (two spans) and p8k at 16384 (one
# batch, pipelined), twice with 512-column spans (the run-to-run floor) and once with 1024; lkl.py: PPL, per-row KL
# against the first run, top-1 agreement.  The service must be stopped (abw.sh probe).
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=2 STRATA_QKV_FUSE=1 GGML_SYCL_HC_POST_V2=1 STRATA_ROPE_INPLACE=1 GGML_SYCL_DSV4_MQA_QV=1 GGML_SYCL_DSV4_MQA_SM=1 STRATA_ATTN_SDPA=1 STRATA_XE_CPU_LIGHT=4 STRATA_XE_PREFETCH_MIN=4096 STRATA_XE_DENSE_LITE=1 STRATA_XE_DENSE_INLINE=1 STRATA_XE_GATHER_INIT=1 STRATA_XE_HOTRING=1 STRATA_XE_HOTRING_N=2"
for P in p2k p8k; do
  PBS=2048; [ $P = p8k ] && PBS=16384
  for R in a b z; do
    N=ls_${P}_$R
    E=""; X=""
    [ $R = z ] && { E="GGML_SYCL_DSV4_SDPA_N=1024"; X="--prefill-chunk 1024"; }
    env $B $E RUN_TIMEOUT=1200 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch $PBS $X --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "^rc=" logs/$N.err)"
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/ls_${P}_a.logits logs/ls_${P}_b.logits logs/ls_${P}_z.logits
  rm -f logs/ls_${P}_*.logits
done
