#!/bin/bash
# tlppl-ltc.sh - STRATA_XE_LT_COMPACT numerics (patch-ltcompact.py: the CPU-light experts' X rows and sums carried as
# compact rows, the combine's CPU part scattered in after it - the same additions in the same order): one-shot
# prompt-path runs of p512 (one batch) and p2k (a pipelined 2048-token batch: no prefetch below 4096 tokens, so its
# calls run light experts on the CPU pool) with the service's dense settings + CPU_LIGHT=4, twice with the old form
# (the run-to-run floor) and once with it; lkl.py: PPL, per-row KL against the first run, top-1 agreement.  The
# service must be stopped (abw.sh probe).
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=2 STRATA_QKV_FUSE=1 GGML_SYCL_HC_POST_V2=1 STRATA_ROPE_INPLACE=1 GGML_SYCL_DSV4_MQA_QV=1 GGML_SYCL_DSV4_MQA_SM=1 STRATA_XE_CPU_LIGHT=4 STRATA_XE_PREFETCH_MIN=4096 STRATA_XE_DENSE_LITE=1 STRATA_XE_DENSE_INLINE=1 STRATA_XE_GATHER_INIT=1"
for P in p512 p2k; do
  for R in a b z; do
    N=lc_${P}_$R
    E="STRATA_XE_LT_COMPACT=0"; [ $R = z ] && E="STRATA_XE_LT_COMPACT=1"
    env $B $E RUN_TIMEOUT=900 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch 2048 --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "^rc=" logs/$N.err)"
    echo "   CPU-light: $(grep -a -h "CPU-light (T" logs/$N.err | tail -1 | cut -c1-160)"
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/lc_${P}_a.logits logs/lc_${P}_b.logits logs/lc_${P}_z.logits
done
rm -f logs/lc_*.logits
