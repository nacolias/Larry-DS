#!/bin/bash
# tattn.sh - STRATA_ATTN_FUSE (ggml_dsv41_kv_fin + ggml_dsv41_attn_dec) on larry: the GPU unit test, decode-path logits
# over p2k.i32 (1-token passes and 5-token passes) fused vs the op chain + KL, then the speed A/B on the essay (plain
# decode and DSpark), service env, route bias 0.5 / 0.3, GPU clocks pinned for the speed runs.  The service must be stopped.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
echo "== unit test $(date +%T)"
timeout 300 ./xolotl/build-xe/test_ds41_attnfuse 2>&1 | tail -60
q() { N=$1; PC=$2; shift 2; env $B "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk $PC $RB --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N] $(grep -h PPL logs/$N.out) $(grep -h "^rc=" logs/$N.err)"; }
kl() { echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-400)"; }
echo "== logits $(date +%T)"
q qa_off1 1 STRATA_ATTN_FUSE=0
cmp -s logs/qb_s1.logits logs/qa_off1.logits && echo "qa_off1 == qb_s1 of 08:36 (bit-identical: the unfused path is unchanged)" || echo "qa_off1 differs from qb_s1"
q qa_on1 1 STRATA_ATTN_FUSE=1
q qa_off5 5 STRATA_ATTN_FUSE=0
q qa_on5 5 STRATA_ATTN_FUSE=1
kl qa_off1 qa_on1; kl qa_off5 qa_on5; kl qa_off1 qa_off5
echo "== speed $(date +%T)"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"; }
for i in 1 2; do run ap_off$i STRATA_ATTN_FUSE=0; run ap_on$i STRATA_ATTN_FUSE=1; done
for i in 1 2; do run ad_off$i $D STRATA_ATTN_FUSE=0; run ad_on$i $D STRATA_ATTN_FUSE=1; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TATTN done $(date +%T)"
