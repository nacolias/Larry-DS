#!/bin/bash
# tattn2.sh - tattn.sh's logits with EXACT routing (--route-bias 0 --route-bias-file 0): with the bias the selection
# depends on which experts happen to be resident (the promotion cache changes run to run), so only exact routing is
# deterministic on the decode path.  STRATA_ATTN_FUSE 0 vs 1 at 1-token and 5-token passes.  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
RX="--route-bias 0 --route-bias-file 0"
q() { N=$1; PC=$2; shift 2; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk $PC $RX --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N] $(grep -h PPL logs/$N.out) $(grep -h "^rc=" logs/$N.err)"; }
kl() { echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-400)"; }
echo "== exact-routing logits $(date +%T)"
q qx_off1 1 STRATA_ATTN_FUSE=0
cmp -s logs/qb_x1.logits logs/qx_off1.logits && echo "qx_off1 == qb_x1 of 08:33 (bit-identical)" || echo "qx_off1 differs from qb_x1"
q qx_on1 1 STRATA_ATTN_FUSE=1
cmp -s logs/qx_off1.logits logs/qx_on1.logits && echo "qx_on1 == qx_off1 (bit-identical)" || kl qx_off1 qx_on1
q qx_off5 5 STRATA_ATTN_FUSE=0
q qx_on5 5 STRATA_ATTN_FUSE=1
cmp -s logs/qx_off5.logits logs/qx_on5.logits && echo "qx_on5 == qx_off5 (bit-identical)" || kl qx_off5 qx_on5
echo "TATTN2 done $(date +%T)"
