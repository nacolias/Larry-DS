#!/bin/bash
# ttp9.sh - engine1 after the lazy KV replicas (STRATA_TP_LAZY_KV) and the cards' own promotion repack
# (STRATA_TP_PC_DEVREPACK), at the service's settings (ctx 262144, 16K batches, FLEX 1 / margin 1.5), clocks pinned:
#   chk   every promoted slot checked against a fresh repack (STRATA_XE_PROMOTE_CHECK: "0 bad")
#   ppl   decode-path PPL / logits of p2k (exact routing; the replicas grow 1024 -> 2048 -> 4096 rows) vs ppl_tp1
#   speed DSpark x3 + plain x2 with TP, today's path x2 + x1 (essay x400, bias 0.5 / 0.3)
#   long  the 95K prefix at ctx 262144 (the replicas grow to 128K rows after the prompt; RESERVE 2.8: the flex region
#         has to give VRAM back) and the 300-token prompt
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() { N=$1; IDS=$2; NT=$3; shift 3; env $B "$@" RUN_TIMEOUT=1800 ./run.sh $N --ids-file $IDS -n $NT --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1
  echo "== $N $(grep -h "^prompt:" logs/$N.err | cut -c9-48) | $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -h "^rc=" logs/$N.err)"
  grep -h "promotion cache: [0-9]* promoted" logs/$N.err | tail -1
  grep -h "^tp: rank 1\|KV replicas now\|flex region .* -> " logs/$N.err | tail -4
  grep -h "card [1-3]:.*VRAM free" logs/$N.err | tr "\n" " "; echo
  grep -h "refused\|OUT_OF\|failed\|differs\|no room\|tp: no VRAM" logs/$N.err | sort | uniq -c | head -4; }
run chk pe.i32 400 STRATA_XE_RESERVE_GIB=3.5 $D $T STRATA_XE_PROMOTE_CHECK=1
( cd bench/engine1 && CASES=tp3 ./tppl.sh > /dev/null 2>&1 ) ; grep -h PPL logs/ppl_tp3.out | tail -1
echo "KL ppl_tp1 || ppl_tp3: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/ppl_tp1.logits logs/ppl_tp3.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"
grep -h "KV replicas now" logs/ppl_tp3.err | tail -3
for i in 1 2 3; do run s9_dtp$i pe.i32 400 STRATA_XE_RESERVE_GIB=3.5 $D $T; [ $i -le 2 ] && run s9_doff$i pe.i32 400 STRATA_XE_RESERVE_GIB=3.5 $D; done
for i in 1 2; do run s9_ptp$i pe.i32 400 STRATA_XE_RESERVE_GIB=3.5 $T; done
run s9_poff1 pe.i32 400 STRATA_XE_RESERVE_GIB=3.5
run s9_ltp p95ke.i32 200 STRATA_XE_RESERVE_GIB=2.8 $D $T
run s9_stp p300.i32 64 STRATA_XE_RESERVE_GIB=3.5 $D $T
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP9 done $(date +%T)"
