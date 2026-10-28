#!/bin/bash
# launch-t160.sh - abwT160 (#3): the service candidate - the fused attention + the expert cards' X in fp16 (FN16=2), the
# indexer all on card 0 (STRATA_LI_REMOTE=0; abwT159: 984K 307.7 s, 131K 39.3 s) - numerics (tlppl-ab.sh: p8k KL, p30k
# PPL) and the 131K + 984K needles with turns; then share 65 again (abwT159's s65 engine went silent after its 984K batch:
# logs kept this time)
cd /opt/ai/dsv41
B=/opt/ai/dsv41/fsabin
setsid nohup bash bench/speed/abw.sh \
  "@XE_BIN=$B LD_LIBRARY_PATH=$B bash bench/speed/tlppl-ab.sh fsafn STRATA_ATTN_FSA=1,STRATA_XE_FN16=2" \
  "@bash bench/longctx/lt1m.sh $B cand:STRATA_ATTN_FSA=1,STRATA_XE_FN16=2,STRATA_LI_REMOTE=0 s65b:STRATA_ATTN_FSA=1,GGML_SYCL_LI_REMOTE_SHARE=65" \
  > logs/abwT160.log 2>&1 < /dev/null &
echo "abwT160 launched: logs/abwT160.log"
