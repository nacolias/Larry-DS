#!/bin/bash
# tr22.sh - (1) the fused Markov chain's exactness: the essay with the old chain + copies and with the new defaults,
# promotions and replicas off (the CPU / card split then static: the same drafts must give the same tokens), 300
# tokens, the token streams compared; (2) the promotions' in-flight cap: math with STRATA_TP_PROMOTE_PAR=16 against
# tr21_math (8); (3) the CPU pool against a streaming read (bench_cpu_pool, CPU only; built here).  The host
# placement of tr20 throughout.  Service held, clocks pinned.
cd /opt/ai/dsv41/bench/engine1
TARGETS="${TR22_TARGETS:-bench_cpu_pool}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
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
  env $B $D $T $H "$@" STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n $n $A > /dev/null 2>&1
  echo "== $N [$*] $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "promotion cache: [0-9]* promoted" | head -1 | cut -c1-160
}
if [ -z "$TR22_NOEXACT" ]; then
  run tr22_x_old pe.i32 300 STRATA_XE_PROMOTE=0 STRATA_XE_REPLICA_GIB=0 STRATA_TP_DRAFT_MK=0 STRATA_TP_DRAFT_PUSH=0
  run tr22_x_new pe.i32 300 STRATA_XE_PROMOTE=0 STRATA_XE_REPLICA_GIB=0
  run tr22_x_old2 pe.i32 300 STRATA_XE_PROMOTE=0 STRATA_XE_REPLICA_GIB=0 STRATA_TP_DRAFT_MK=0 STRATA_TP_DRAFT_PUSH=0
  for p in "tr22_x_old tr22_x_new" "tr22_x_old tr22_x_old2"; do set -- $p
    cmp -s logs/$1.out logs/$2.out && echo "$1 vs $2: token streams IDENTICAL ($(wc -l < logs/$2.out) tokens)" || echo "$1 vs $2: differ at $(cmp logs/$1.out logs/$2.out | grep -o 'line [0-9]*')"
  done
fi
run tr22_math_par16 pmath.i32 1000 $REP STRATA_TP_PROMOTE_PAR=16
python3 -I bench/engine1/cyc.py logs/tr21_math.cpu logs/tr22_math_par16.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "== bench_cpu_pool (6 workers + host on CPU 0; 7 workers; nt 2)"
xolotl/build-xe/bench_cpu_pool --workers 6 --host-cpu 0 --gib 5 --iters 40 2>&1 | tail -4
xolotl/build-xe/bench_cpu_pool --workers 7 --host-cpu 0 --gib 5 --iters 40 2>&1 | tail -3
xolotl/build-xe/bench_cpu_pool --workers 6 --host-cpu 0 --gib 5 --iters 40 --nt 2 2>&1 | tail -3
dmesg -T | grep -c "CAT error"
echo "TR22 done $(date +%T)"
