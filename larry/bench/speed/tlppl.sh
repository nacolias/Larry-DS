#!/bin/bash
# tlppl.sh - STRATA_XE_CPU_LIGHT's numerics: one-shot prompt-path runs (the service's tier settings, the prompt as one
# batch, the prefetch off so every call can take the light path) of p512 (one batch of 512: not pipelined) and p2k (a
# pipelined batch: X read back off the dense card) with T = 0, 0 again (the run-to-run floor) and T = TL_T (4);
# lkl.py: PPL, per-row KL against the first T = 0 run, top-1 agreement.  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_PREFETCH_MIN=1000000 STRATA_XE_PROF=1"
T=${TL_T:-4}
for P in p512 p2k; do
  for R in 0 0b $T; do
    N=lp_${P}_$R
    env $B STRATA_XE_CPU_LIGHT=${R%b} RUN_TIMEOUT=900 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch 2048 --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "^rc=" logs/$N.err)"
    grep -a "CPU-light" logs/$N.err | tail -1
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/lp_${P}_0.logits logs/lp_${P}_0b.logits logs/lp_${P}_$T.logits
done
rm -f logs/lp_*.logits
