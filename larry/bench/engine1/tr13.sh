#!/bin/bash
# tr13.sh - where a DSpark cycle's time goes now (no build): the service's settings with the replicas, clocks pinned.
#   (a) essay / code / math, 1000 tokens, STRATA_DSPARK_PROF=1 (the draft step's host phases) + the CPU log (cycle by
#       width: cyc.py);
#   (b) the essay with every layer's stamps (the cards' waits with the replicas on);
#   (c) the essay under unitrace (-n 400, 3 s of decode): the draft's kernels on card 0 (dtrace.py; the TP passes are
#       replayed graphs and do not show).
# The service must be held down.
cd /opt/ai/dsv41
[ -f /run/lock/larry-profile-a.hold ] || { echo "no hold file"; exit 1; }
systemctl is-active -q larry-ds41 && { echo "the service is running"; exit 1; }
pgrep -x ds41_generate_x > /dev/null && { echo "an engine is running"; exit 1; }
source xe-env.sh
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_MEM_FLOOR_GIB=16 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_ENGRAM_THREADS=256 GGML_SYCL_LI_GEMM_MIN=1 GGML_SYCL_LI_XMX=1 STRATA_XE_JOBNT=1 STRATA_XE_COMBINE=1 STRATA_QKV_FUSE=1 STRATA_XE_FLEX_GIB=1 STRATA_XE_FLEX_MARGIN_GIB=1.5"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
T="STRATA_TP=4 STRATA_TP_CACHE=/srv/dsv41/tpcache STRATA_TP_PAGE=1 STRATA_TP_PROF=1"
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
A="--ctx 262144 --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3"
/usr/local/sbin/larry-gpu-clock pin > /dev/null
for c in ${TR13_CASES:-essay code math essay_st}; do
  case $c in essay*) F=pe.i32 ;; code*) F=pcode.i32 ;; math*) F=pmath.i32 ;; esac
  case $c in *_st) X="STRATA_TP_STAMP_LAYER=-1" ;; *) X="STRATA_DSPARK_PROF=1" ;; esac
  N=tr13_$c
  env $B $D $T $REP $X STRATA_XE_CPU_LOG=logs/$N.cpu RUN_TIMEOUT=1800 ./run.sh $N --ids-file $F -n 1000 $A > /dev/null 2>&1
  echo "== $N $(grep -ah "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s\|[0-9]* cpu /" | tr "\n" " ") | $(grep -ah "DSpark: [0-9]* verify" logs/$N.err | grep -o "verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass\|[0-9.]* tokens/pass" | tr "\n" " ") $(grep -ah "^rc=" logs/$N.err)"
  tr "\r" "\n" < logs/$N.err | grep -a "dspark prof" | tail -1
  tr "\r" "\n" < logs/$N.err | grep -a "tp stamps all layers" | tail -1
done
python3 -I bench/engine1/cyc.py logs/tr13_essay.cpu logs/tr13_code.cpu logs/tr13_math.cpu logs/tr11_essay_um.cpu
if [ -z "$TR13_NOTRACE" ]; then
  env $B $D $T $REP RUN_TIMEOUT=900 ./utrace.sh tr13ut 3 --ids-file pe.i32 -n 400 $A > /dev/null 2>&1
  echo "== tr13ut $(grep -ah "^decode:" logs/tr13ut.err | grep -o "= [0-9.]* tok/s") $(grep -ah "^rc=" logs/tr13ut.err) $(du -sh logs/ut-tr13ut | cut -f1)"
  python3 -I bench/engine1/dtrace.py logs/ut-tr13ut 1500 30
fi
/usr/local/sbin/larry-gpu-clock restore > /dev/null
dmesg -T | grep -c "CAT error"
echo "TR13 done $(date +%T)"
