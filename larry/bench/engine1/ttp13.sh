#!/bin/bash
# ttp13.sh - engine1 + the pager (STRATA_TP_PAGE: card 0's copies of the TP cards' dense weights paged in for the
# prompt path only) at the service config (ctx 262144, 16K batches, FLEX 1 / margin 1.5, RESERVE 3.5), clocks pinned:
#   chk   STRATA_TP_PAGE_CHECK=1: every unit refilled from the shards and compared with the load (bit for bit)
#   ppl   the PROMPT path's logits (p2k as one 2048-token layer-major batch, exact routing) paged vs resident
#   speed DSpark essay x2, plain x1; prompts of 300, 20K and 95K tokens (the fills ride every batch's layers)
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
RB="--route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
show() { N=$1
  echo "== $N $(grep -ah "^prompt:" logs/$N.err | cut -c9-60) | $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "tp pager\|dense card's expert cap\|dense card 0 seeded\|VRAM: .* free after the dense half\|card 0 (dense)" | cut -c1-200 | head -8
  tr "\r" "\n" < logs/$N.err | grep -a "refused\|OUT_OF\|failed\|differs\|tp: no VRAM\|abort\|terminate\|missing" | sort | uniq -c | head -4; }
run() { N=$1; IDS=$2; NT=$3; shift 3; env $B "$@" RUN_TIMEOUT=1800 ./run.sh $N --ids-file $IDS -n $NT --ctx 262144 --prompt-batch 16384 $RB > /dev/null 2>&1; show $N; }
run t13_chk pe.i32 400 $D $T STRATA_TP_PAGE_CHECK=1
grep -aq "check ok" logs/t13_chk.err || { echo "CHECK NOT OK - stopping"; /usr/local/sbin/larry-gpu-clock restore > /dev/null; exit 1; }
for m in on off; do
  P=""; [ $m = off ] && P="STRATA_TP_PAGE=0"
  env $B $T $P RUN_TIMEOUT=1800 ./run.sh t13_ppl_$m --ids-file p2k.i32 -n 1 --ppl --ctx 8192 --prompt-batch 2048 --route-bias 0 --route-bias-file 0 --dump-logits logs/t13_ppl_$m.logits > /dev/null 2>&1
  echo "[t13_ppl_$m] $(grep -ah PPL logs/t13_ppl_$m.out | tail -1) $(grep -ah "^rc=" logs/t13_ppl_$m.err)"
done
cmp -s logs/t13_ppl_on.logits logs/t13_ppl_off.logits && echo "prompt-path logits paged == resident (bit-identical)" || echo "KL on || off: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/t13_ppl_on.logits logs/t13_ppl_off.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-300)"
for i in 1 2; do run t13_dtp$i pe.i32 400 $D $T; done
run t13_ptp pe.i32 400 $T
run t13_stp p300.i32 64 $D $T
run t13_mtp p20k.i32 200 $D $T
run t13_ltp p95ke.i32 200 $D $T
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP13 done $(date +%T)"
