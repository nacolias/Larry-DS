#!/bin/bash
# qv_engine.sh - our engine (the service's settings) over the same qv/corpus sequences, full logits per row:
#   pp  : the prompt path (how the service reads every prompt; routing exact)  - all 18 sequences
#   dx  : the decode path (--prefill-chunk 1: every row a decode pass), exact routing   - 2 sequences a domain
#   ds  : the decode path with the service's routing bias (--route-bias 0.5 --route-bias-file 0.3) - the same 10
# -> /srv/dsv41/qv-work/elogits/{pp,dx,ds}_sNN.logits (f32 [2048, V]).  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh   # (oneAPI's vars.sh reads unset variables: no set -u before it)
set -u
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
O=/srv/dsv41/qv-work/elogits; mkdir -p $O
C=/opt/ai/dsv41/qv/corpus
one() { tag=$1; i=$2; shift 2; N=qv_${tag}_s$i
  [ -s $O/${tag}_s$i.logits ] && return
  env $B RUN_TIMEOUT=900 ./run.sh $N --ids-file $C/s$i.i32 -n 1 --ppl "$@" --dump-logits $O/${tag}_s$i.logits > /dev/null 2>&1
  echo "$tag s$i: $(grep -h PPL logs/$N.out) $(grep -h "^rc=" logs/$N.err)"; }
for i in $(seq -w 0 17); do one pp $i --prompt-batch 2048; done
SUB="00 01 04 05 07 08 10 11 14 16"
for i in $SUB; do
  one dx $i --prompt-batch 0 --prefill-chunk 1 --route-bias 0 --route-bias-file 0
  one ds $i --prompt-batch 0 --prefill-chunk 1 --route-bias 0.5 --route-bias-file 0.3
done
echo "ENGINE DONE $(date +%T)"
