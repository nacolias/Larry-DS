#!/bin/bash
# tbias2.sh - tbias.sh's follow-up: Xolotl's "quality-first" bias 0.25 / 0.15 (speed x2 with DSpark, decode-path PPL +
# logits for KL vs exact routing (logs/qb_x1.logits from tbias.sh)).  Service env, clocks pinned, service stopped.
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RQ="--route-bias 0.25 --route-bias-file 0.15"
for i in 1 2; do N=bd_q$i
  env $B $D RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RQ > /dev/null 2>&1
  echo "== $N [q: $RQ] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|experts [0-9]* hit / [0-9]* cpu\|[0-9.]*% VRAM hits" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
N=qb_q1
env $B RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 $RQ --dump-logits logs/$N.logits > /dev/null 2>&1
echo "[$N q] $(grep -h PPL logs/$N.out) | $(grep -h "^prompt:" logs/$N.err | grep -o "experts [0-9]* hit / [0-9]* cpu") $(grep -h "^rc=" logs/$N.err)"
echo "KL qb_x1 || qb_q1: $(/opt/ai/dsv41/venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/qb_x1.logits logs/qb_q1.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-400)"
echo "TBIAS2 done $(date +%T)"
