#!/bin/bash
# ttp14.sh - engine1's pager: the prompt path's logits (p2k as one 2048-token layer-major batch, exact routing) paged
# vs resident, each twice (the batch path's own run-to-run floor first), and an essay DSpark run (the release wait:
# card 0's free VRAM measured after the pages are back); clocks pinned
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PROF=1"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in off1 off2 on1 on2; do
  P=""; case $c in off*) P="STRATA_TP_PAGE=0" ;; esac
  env $B $T $P RUN_TIMEOUT=1800 ./run.sh t14_$c --ids-file p2k.i32 -n 1 --ppl --ctx 8192 --prompt-batch 2048 --route-bias 0 --route-bias-file 0 --dump-logits logs/t14_$c.logits > /dev/null 2>&1
  echo "[t14_$c] $(grep -ah PPL logs/t14_$c.out | tail -1) $(grep -ah "^rc=" logs/t14_$c.err)"
done
kl() { cmp -s logs/$1.logits logs/$2.logits && echo "$1 == $2 (bit-identical)" || echo "KL $1 || $2: $(venv/bin/python -I xolotl/tools/v41/logits_kl.py logs/$1.logits logs/$2.logits --ids p2k.i32 2>&1 | tr "\n" " " | cut -c1-260)"; }
kl t14_off1 t14_off2; kl t14_on1 t14_on2; kl t14_off1 t14_on1; kl t14_off2 t14_on2
env $B $D $T RUN_TIMEOUT=1800 ./run.sh t14_dtp --ids-file pe.i32 -n 400 --ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 > /dev/null 2>&1
echo "== t14_dtp $(grep -ah "^decode:" logs/t14_dtp.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/t14_dtp.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/t14_dtp.err)"
tr "\r" "\n" < logs/t14_dtp.err | grep -a "tp pager\|VRAM: .* free after the dense half\|dense card 0 seeded" | cut -c1-200
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "TTP14 done $(date +%T)"
