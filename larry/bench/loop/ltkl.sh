#!/bin/bash
# ltkl.sh NAME - long-context numerics of the 2026-10-28 prompt-path changes, after the 2026-10-29 agent session's
# thinking loops at 56K-155K: the service's engine as it runs (a), with those changes switched off (b: the attention
# kernel STRATA_ATTN_FSA, the expert cards' fp16 X STRATA_XE_FN16, xe_dqgemm v2, the indexer's fused top-k and its two
# stages, the replicas' read-ahead), as it runs again (a2: the run-to-run floor).  ltkl.py per config: fresh 96K and
# 150K prompts of code, their last 2,048 rows' logits dumped, 192 tokens greedy with the TP replica check; then
# tailkl.py a against a2 and b.  The service must be stopped: run as an abw.sh probe ("@bash bench/loop/ltkl.sh NAME").
set -u
cd /opt/ai/dsv41
PY=/opt/ai/dsv41/venv/bin/python
OLD="STRATA_ATTN_FSA=0 STRATA_XE_FN16=0 STRATA_XE_DQ2=0 STRATA_LI_TOPK=0 GGML_SYCL_TOPK2=0 STRATA_TP_AHEAD_S=0"
for R in a b a2; do
  E=""
  [ $R = b ] && E="$OLD"
  echo "=== $1_$R $(date +%T)"
  LT_ENV="$E" $PY -I bench/loop/ltkl.py ${1}_$R || echo "run $R FAILED"
  grep -h -E "^serve: prompt [0-9]+ tokens|tp sync check: conv" logs/engine-lt-${1}_$R.log | cut -c1-150
done
echo "=== tail KL $(date +%T)"
$PY -I bench/loop/tailkl.py logs/lt-${1}_a logs/lt-${1}_a2 logs/lt-${1}_b
echo "=== done $(date +%T)"
