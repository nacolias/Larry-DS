#!/bin/bash
# ttp2.sh - STRATA_TP_OPROJ=2 (attention graph in two halves, the output projection split over the four cards between
# them) with / without the kept graphs replayed (GGML_SYCL_GRAPH_CACHE): decode-path perplexity, then plain decode and
# DSpark (essay) with the host profiler + timeline.  Every run with the same expert-card reserve (4.2 GiB: room for
# the split's 620 MB a card).
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=4.2 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0"
D="STRATA_DSPARK=$SIDECAR STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.3 STRATA_DSPARK_DENSE_TYPE=q6_K"
N=tq2
env $B STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=1 RUN_TIMEOUT=900 ./run.sh $N --ids-file p2k.i32 -n 1 --ppl --prompt-batch 0 --prefill-chunk 1 > /dev/null 2>&1
echo "[ppl TP_OPROJ=2 GRAPH_CACHE=1] $(grep -h PPL logs/$N.out) | $(grep -h "attention output split" logs/$N.err | sort | uniq -c | tr "\n" " ") $(grep -h "not split\|failed\|Abort\|abort" logs/$N.err | sort | uniq -c | head -3 | tr "\n" " ")"
run() { N=$1; shift; env $B STRATA_DEC_PROF=1 STRATA_GPU_TL=1 "$@" RUN_TIMEOUT=900 ./run.sh $N --ids-file pe.i32 -n 400 > /dev/null 2>&1
  echo "== $N ($(echo "$*" | sed "s|$SIDECAR|sidecar|")) $(grep -h "^decode:" logs/$N.err | grep -o "= [0-9.]* tok/s") | $(grep -h "DSpark: [0-9]* verify" logs/$N.err | grep -o "[0-9.]* tokens/pass\|verify [0-9.]* ms/pass\|draft [0-9.]* ms/pass" | tr "\n" " ")"
  grep -h "dec prof" logs/$N.err | tail -1
  grep -h "gpu timeline" logs/$N.err | tail -1 | tr ";" "\n" | sed -n "1,3p"
  grep -h "attention output split\|not split\|failed" logs/$N.err | sort | uniq -c | head -4; }
run tq0g0 STRATA_TP_OPROJ=0 GGML_SYCL_GRAPH_CACHE=0
run tq0g1 STRATA_TP_OPROJ=0 GGML_SYCL_GRAPH_CACHE=1
run tq2g0 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=0
run tq2g1 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=1
run tqs0g1 STRATA_TP_OPROJ=0 GGML_SYCL_GRAPH_CACHE=1 $D
run tqs2g1 STRATA_TP_OPROJ=2 GGML_SYCL_GRAPH_CACHE=1 $D
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
