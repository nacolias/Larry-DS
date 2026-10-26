#!/bin/bash
# tlppl-glue.sh - the dense-span glue kernels' numerics (patch-glue1/2, patch-ropeip): one-shot prompt-path runs of p512
# (one batch, not pipelined) and p2k (a pipelined 2048-token batch) - the service's settings, twice without them (the
# run-to-run floor) and once with them; lkl.py: PPL, per-row KL against the first run, top-1 agreement.  The service
# must be stopped (abw.sh probe).
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
G="GGML_SYCL_HC_POST_V2=1 STRATA_ROPE_INPLACE=1 GGML_SYCL_DSV4_MQA_QV=1 GGML_SYCL_DSV4_MQA_SM=1"
for P in p512 p2k; do
  for R in a b g; do
    N=lg_${P}_$R
    E=""; [ $R = g ] && E="$G"
    env $B $E RUN_TIMEOUT=900 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch 2048 --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "^rc=" logs/$N.err)"
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/lg_${P}_a.logits logs/lg_${P}_b.logits logs/lg_${P}_g.logits
done
rm -f logs/lg_*.logits
