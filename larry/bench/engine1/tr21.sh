#!/bin/bash
# tr21.sh - the draft's Markov chain fused (STRATA_TP_DRAFT_MK) and the draft's sums pushed (STRATA_TP_DRAFT_PUSH):
# rebuild libds41_tp, then with the host placement (STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6
# STRATA_XE_HELPER_CPUS=7,15): the essay with the old chain + copies and with the fused chain + pushes, 400 tokens
# (the same drafts expected: the output token streams compared), the draft's phase stamps; then math and essay at
# 1000 tokens with the new defaults (cyc.py against tr20's).  Service held, clocks pinned.
cd /opt/ai/dsv41/bench/engine1
TARGETS="${TR21_TARGETS:-ds41_tp}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1 STRATA_TP_DRAFT=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
H="STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6 STRATA_XE_HELPER_CPUS=7,15"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
run() {   # NAME FILE TOKENS ENV...
  local N=$1 F=$2 n=$3; shift 3
  env $B $D $T $REP $H "$@" STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n $n $A > /dev/null 2>&1
  echo "== $N [$*] $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "^tp draft:" | grep -o "sums by [A-Za-z0-9 ]*" | head -1
  tr "\r" "\n" < logs/$N.err | grep -a "tp draft stamps card [03]" | tail -2
}
run tr21_ess_old pe.i32 400 STRATA_TP_DRAFT_MK=0 STRATA_TP_DRAFT_PUSH=0 STRATA_TP_DRAFT_STAMP=1
run tr21_ess_new pe.i32 400 STRATA_TP_DRAFT_STAMP=1
cmp -s logs/tr21_ess_old.out logs/tr21_ess_new.out && echo "token streams IDENTICAL ($(wc -l < logs/tr21_ess_new.out) tokens)" || echo "token streams differ from line $(cmp logs/tr21_ess_old.out logs/tr21_ess_new.out | grep -o 'line [0-9]*')"
run tr21_math pmath.i32 1000
run tr21_essay pe.i32 1000
python3 -I bench/engine1/cyc.py logs/tr20_math_h0w6.cpu logs/tr21_math.cpu
python3 -I bench/engine1/cyc.py logs/tr20_essay_h0w6.cpu logs/tr21_essay.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR21 done $(date +%T)"
