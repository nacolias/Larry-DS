#!/bin/bash
# tbias.sh - the decode routing bias on larry: exact (--route-bias 0 --route-bias-file 0) vs the service's 0.5 / 0.3 vs
# the engine default 1.0 / 0.6 (what one-shot runs without the flags use).  The bias is selection-only and applies to
# decode / verify rows only (prompts route exactly).  Service env, GPU clocks pinned; the service must be stopped.
#   speed:   the essay (pe.i32 -n 400), DSpark x3 (x2 at 1.0) and plain x2 per setting, interleaved
#   quality: decode-path PPL over p2k.i32 (--prefill-chunk 1: every row a decode pass, bias on) + full logits dumps
#            (logs/qb_*.logits) for KL between settings; exact twice = the run-to-run floor
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
declare -A RB=([x]="--route-bias 0 --route-bias-file 0" [s]="--route-bias 0.5 --route-bias-file 0.3" [d]="--route-bias 1.0 --route-bias-file 0.6")
# run NAME BIASKEY ENV... : the essay
run() { N=$1; K=$2; shift 2; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 ${RB[$K]} > /dev/null 2>&1
  echo "== $N [$K: ${RB[$K]}] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|experts [0-9]* hit / [0-9]* cpu\|[0-9.]*% VRAM hits" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"; }
for i in 1 2 3; do
  run bd_x$i x $D; run bd_s$i s $D
  [ $i -lt 3 ] && run bd_d$i d $D
done
for i in 1 2; do run bp_x$i x; run bp_s$i s; done
# quality: decode-path rows with the bias on
q() { N=$1; K=$2; env $B RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 ${RB[$K]} --dump-logits logs/$N.logits > /dev/null 2>&1
  echo "[$N $K] $(grep -h PPL logs/$N.out) | $(grep -h "^prompt:" logs/$N.err | grep -o "experts [0-9]* hit / [0-9]* cpu") $(grep -h "^rc=" logs/$N.err)"; }
q qb_x1 x; q qb_s1 s; q qb_d1 d; q qb_x2 x
/usr/local/sbin/larry-gpu-clock restore > /dev/null
for p in "qb_x1 qb_x2" "qb_x1 qb_s1" "qb_x1 qb_d1"; do
  set -- $p
  echo "KL $1 || $2: $(/opt/ai/dsv41/venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-400)"
done
echo "TBIAS done $(date +%T)"
