#!/bin/bash
# launch-t159.sh - abwT159 (#3): with the fused attention card 0 is ~0.9 ms a span faster, and at >= 131K the expert cards
# also score 65 % of the indexer's queries (GGML_SYCL_LI_REMOTE_SHARE=35 % stays on card 0, tuned without it) - abwT158's
# 984K needle got slower (339 s against 308: card 0 -28 s, the tier waits +52 s).  The share again: 131K + 984K needles
# (bench/longctx/lt1m.sh, fsabin, STRATA_ATTN_FSA=1) at card 0 keeping 50 / 65 %, and without the remote share.
cd /opt/ai/dsv41
B=/opt/ai/dsv41/fsabin
setsid nohup bash bench/speed/abw.sh \
  "@bash bench/longctx/lt1m.sh $B s50:STRATA_ATTN_FSA=1,GGML_SYCL_LI_REMOTE_SHARE=50 s65:STRATA_ATTN_FSA=1,GGML_SYCL_LI_REMOTE_SHARE=65 r0:STRATA_ATTN_FSA=1,STRATA_LI_REMOTE=0" \
  > logs/abwT159.log 2>&1 < /dev/null &
echo "abwT159 launched: logs/abwT159.log"
