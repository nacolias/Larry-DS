#!/bin/bash
# tr20.sh - the CPU pool's place on the host: its 7 workers own CPUs 1-7, everything else floated (the TP pass's host
# loop, the promotion fill helpers, the Engram reader) and lands on the workers' SMT siblings.  Rebuild the tier + TP
# libs (STRATA_HOST_CPUS / STRATA_XE_HELPER_CPUS / STRATA_XE_POOL_WORKERS, xe_cpus.hpp); then math (the CPU pool's
# text) at the service's settings with the TP draft: base / host on CPU 0 / + helpers on CPU 8 / 6 workers + helpers
# on core 7 / host on CPU 0 + the card repack (STRATA_TP_PC_DEVREPACK=1); the essay for base and TR20_BEST.  1000
# tokens, clocks pinned, the CPU log (cyc.py; the CPU layers' GB/s).  Service held.
cd /opt/ai/dsv41/bench/engine1
TARGETS="${TR20_TARGETS:-ds41_xe ds41_tp}" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
cd /opt/ai/dsv41
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1 STRATA_TP_DRAFT=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
cfg() {
  case $1 in
    base) echo "" ;;
    h0) echo "STRATA_HOST_CPUS=0" ;;
    h0k8) echo "STRATA_HOST_CPUS=0 STRATA_XE_HELPER_CPUS=8" ;;
    h0w6) echo "STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6 STRATA_XE_HELPER_CPUS=7,15" ;;
    h0dev) echo "STRATA_HOST_CPUS=0 STRATA_TP_PC_DEVREPACK=1" ;;
    h0w6dev) echo "STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6 STRATA_XE_HELPER_CPUS=7,15 STRATA_TP_PC_DEVREPACK=1" ;;
  esac
}
rate() {   # the CPU layers' GB/s (one expert / several) and the one-expert layer's us
  python3 -I - "$1" << 'PY'
import sys
L = [l.split() for l in open(sys.argv[1]) if l.startswith("L ")]
one = [x for x in L if int(x[4]) == 1]; mul = [x for x in L if int(x[4]) > 1]
r = lambda v: sum(int(x[5]) for x in v) / max(1e-9, sum(float(x[6]) for x in v)) / 1e3
print("   CPU layers %d: one expert %.0f us (%.1f GB/s), several %.1f GB/s" % (len(L), sum(float(x[6]) for x in one) / max(1, len(one)), r(one), r(mul)))
PY
}
for c in ${TR20_CASES:-math_base math_h0 math_h0k8 math_h0w6 math_h0dev essay_base essay_${TR20_BEST:-h0w6}}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  N=tr20_$c
  env $B $D $T $REP $(cfg ${c#*_}) STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 $A > /dev/null 2>&1
  echo "== $N [$(cfg ${c#*_})] $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  rate logs/$N.cpu
done
python3 -I bench/engine1/cyc.py logs/tr20_math_*.cpu
python3 -I bench/engine1/cyc.py logs/tr20_essay_*.cpu logs/tr19_essay_zc.cpu
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR20 done $(date +%T)"
