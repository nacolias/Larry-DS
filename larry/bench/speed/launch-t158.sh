#!/bin/bash
# launch-t158.sh - abwT158 (#3): the fused prompt attention (fsabin, STRATA_ATTN_FSA=1) through a 131K + 984K needle with
# their decodes and 12K turns (bench/longctx/lt1m.sh: the 1M test), then the expert cards' X in fp16 again (STRATA_XE_FN16
# 1 / 2) now that card 0 is faster: fresh 16K / 31K / 64K prompts; then xe_dqgemm with taller tiles (bench/dqb/dqmt.sh)
# and the fused attention's softmax by transpose-reduce (v6, bench/fa/run_fsa.sh)
cd /opt/ai/dsv41
B=/opt/ai/dsv41/fsabin
AB_PROMPTS=16000,31000,64000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/longctx/lt1m.sh $B fsa1:STRATA_ATTN_FSA=1" \
  fn0:prompts:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_ATTN_FSA=1 \
  fn1:prompts:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_ATTN_FSA=1,STRATA_XE_FN16=1 \
  fn2:prompts:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_ATTN_FSA=1,STRATA_XE_FN16=2 \
  "@bash bench/dqb/dqmt.sh run" "@bash bench/fa/run_fsa.sh" > logs/abwT158.log 2>&1 < /dev/null &
echo "abwT158 launched: logs/abwT158.log"
