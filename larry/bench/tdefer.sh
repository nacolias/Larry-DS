#!/bin/bash
# tdefer.sh - STRATA_FIN_DEFER (the finish inside the next layer's attention hc) on vs off: plain x2 and DSpark x2 each
# (clocks pinned; DSpark's taps read the state after layers 36-38 - a missing flush would show as a collapsed draft
# acceptance), then decode-path logits (exact routing) vs the unfused qd_a.
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; shift; env $B "$@" RUN_TIMEOUT=1500 ./run.sh $N --ids-file pe.i32 -n 400 $RB > /dev/null 2>&1
  echo "== $N [$*] $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)" | sed "s|STRATA_DSPARK=[^ ]* STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K|DSpark|"; }
for i in 1 2; do run dp_on$i; run dp_off$i STRATA_FIN_DEFER=0; done
for i in 1 2; do run dd_on$i $D; run dd_off$i $D STRATA_FIN_DEFER=0; done
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "-- text (DSpark, deferral on / off): $(head -c 160 logs/dd_on1.out | tr "\n" " ")"
echo "--                                 : $(head -c 160 logs/dd_off1.out | tr "\n" " ")"
env $B RUN_TIMEOUT=900 ./run.sh qg_all --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 --route-bias 0 --route-bias-file 0 --dump-logits logs/qg_all.logits > /dev/null 2>&1
echo "[qg_all] $(grep -h PPL logs/qg_all.out) $(grep -h "^rc=" logs/qg_all.err)"
echo "KL qd_a || qg_all: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/qd_a.logits logs/qg_all.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"
echo "TDEFER done $(date +%T)"
