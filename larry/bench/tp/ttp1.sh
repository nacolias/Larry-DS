#!/bin/bash
# ttp1.sh - STRATA_TP_OPROJ (the attention output split over the four cards) A/B: decode-path perplexity (2048 tokens,
# one a step) for correctness, then plain decode and DSpark (essay, 300 / 400 tokens) with the host profiler + timeline
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
for t in 1 0; do
  N=tpp$t
  env $B STRATA_TP_OPROJ=$t RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
  echo "[ppl TP_OPROJ=$t] $(grep -h PPL logs/$N.out) | $(grep -h "attention output split\|oproj" logs/$N.err | sort | uniq -c | head -4 | tr "\n" " ")"
done
run() { N=$1; shift; env $B STRATA_DEC_PROF=1 STRATA_GPU_TL=1 "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($(echo "$*" | sed "s|$SIDECAR|sidecar|")) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ")"
  grep -h "dec prof" logs/$N.err | tail -1
  grep -h "gpu timeline" logs/$N.err | tail -1 | tr ";" "\n" | sed -n "1,3p"
  grep -h "attention output split\|oproj" logs/$N.err | sort | uniq -c | head -4; }
run tpd0 STRATA_TP_OPROJ=0
run tpd1 STRATA_TP_OPROJ=1
run tps0 STRATA_TP_OPROJ=0 $D
run tps1 STRATA_TP_OPROJ=1 $D
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
