#!/bin/bash
# tfuse2.sh - the decode fusion round: unit tests (parts 1-3), then plain-decode and DSpark speed per stage, clocks pinned,
# service env + bias 0.5 / 0.3.  Stages (each env switch defaults on):
#   off  STRATA_ATTN_FUSE=0 STRATA_DIRECT_OUT=0      (everything below needs the fused attention)
#   A    the attention core (kv_fin + attn_dec)       STRATA_Q8_FUSE=0 STRATA_LI_FUSE=0 STRATA_DIRECT_OUT=0
#   AB   + producers' int8 (q8) for the Q6_K products STRATA_LI_FUSE=0 STRATA_DIRECT_OUT=0
#   ABC  + outputs written in place                    STRATA_LI_FUSE=0
#   all  + the fused indexer
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
declare -A ST=([off]="STRATA_ATTN_FUSE=0 STRATA_DIRECT_OUT=0" [A]="STRATA_Q8_FUSE=0 STRATA_LI_FUSE=0 STRATA_DIRECT_OUT=0" [AB]="STRATA_LI_FUSE=0 STRATA_DIRECT_OUT=0" [ABC]="STRATA_LI_FUSE=0" [all]="")
echo "== unit test $(date +%T)"
GGML_SYCL_LI_XMX=1 GGML_SYCL_LI_GEMM_MIN=1 timeout 600 ./xolotl/build-xe/test_ds41_attnfuse 2>&1 | grep -v "^|\|^SYCL\|^Found\|Build with\|GGML_SYCL\|Running with" | grep -E "BAD|NON-FINITE|DIFFER|composite|q8 n=|li mode" | head -60
echo "== speed $(date +%T)"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; S=$2; shift 2; env $B ${ST[$S]} "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$S] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"; }
for s in off A AB ABC all; do run fp_$s $s; done
for s in all off; do run fp2_$s $s; done
for i in 1 2 3; do run fd_off$i off $D; run fd_all$i all $D; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "== dec prof $(date +%T)"
for s in off all; do
  N=fq_$s
  env $B ${ST[$s]} STRATA_DEC_PROF=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 200 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "^dec prof" logs/$N.err | tail -1) $(grep -h "^rc=" logs/$N.err)"
done
echo "== logits (exact routing, vs the unfused qd_a) $(date +%T)"
env $B RUN_TIMEOUT=900 ./run.sh qf_all --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 --route-bias 0 --route-bias-file 0 --dump-logits logs/qf_all.logits > /dev/null 2>&1
echo "[qf_all] $(grep -h PPL logs/qf_all.out) $(grep -h "^rc=" logs/qf_all.err)"
echo "KL qd_a || qf_all: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/qd_a.logits logs/qf_all.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"
echo "TFUSE2 done $(date +%T)"
