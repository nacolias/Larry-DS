#!/bin/bash
# tlppl-sdpa.sh - STRATA_XE_ZCOPY=2 numerics (patch-zcopy2.py + patch-zcopy3.py: every call staged per call, the GEMM tables per round): one-shot prompt-path runs of p512
# (one batch, not pipelined) and p2k (a pipelined 2048-token batch) with the service's dense settings, twice without it
# (the run-to-run floor) and once with it; lkl.py: PPL, per-row KL against the first run, top-1 agreement.  The service
# must be stopped (abw.sh probe).
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 GGML_SYCL_HC_POST_V2=1 STRATA_ROPE_INPLACE=1 GGML_SYCL_DSV4_MQA_QV=1 GGML_SYCL_DSV4_MQA_SM=1"
G="STRATA_XE_ZCOPY=2"
for P in p512 p2k; do
  for R in a b z; do
    N=lz_${P}_$R
    E=""; [ $R = z ] && E="$G"
    env $B $E RUN_TIMEOUT=900 ./run.sh $N --ids-file $P.i32 -n 1 --ppl --prompt-batch 2048 --dump-logits logs/$N.logits
    echo "$N: $(grep -h PPL logs/$N.out) $(grep -h "^prompt:" logs/$N.err | cut -c1-90) $(grep -h "^rc=" logs/$N.err)"
    grep -a -h -c "overrun" logs/$N.err
  done
  /opt/ai/dsv41/venv/bin/python -I bench/speed/lkl.py $P.i32 logs/lz_${P}_a.logits logs/lz_${P}_b.logits logs/lz_${P}_z.logits
done
rm -f logs/lz_*.logits
