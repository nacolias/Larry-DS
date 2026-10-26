#!/bin/bash
# tdet2.sh - is the decode path (exact routing) still run-to-run deterministic?  The same unfused run twice (+ promotion
# off twice: the slot cache moves experts between the CPU pool and the cards asynchronously).
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
RX="--route-bias 0 --route-bias-file 0"
q() { N=$1; shift; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 $RX --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N] $(grep -h PPL logs/$N.out) $(grep -h "^rc=" logs/$N.err)"; }
kl() { cmp -s logs/$1.logits logs/$2.logits && echo "$1 == $2 (bit-identical)" || echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"; }
q qd_a STRATA_ATTN_FUSE=0
kl qx_off1 qd_a
q qd_b STRATA_ATTN_FUSE=0 STRATA_XE_PROMOTE=0
q qd_c STRATA_ATTN_FUSE=0 STRATA_XE_PROMOTE=0
kl qd_b qd_c
echo "TDET2 done $(date +%T)"
