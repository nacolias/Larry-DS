#!/bin/bash
# launch-t157.sh - abwT157 (#3): the fused prompt attention in the engine (staged fsabin, patch-fsa.py, STRATA_ATTN_FSA):
# numerics (tlppl-ab.sh: p8k KL against the run-to-run floor, p30k PPL), then fresh 16K / 31K / 64K prompts and a 131K
# needle with the switch off and on (same binary)
cd /opt/ai/dsv41
B=/opt/ai/dsv41/fsabin
AB_PROMPTS=16000,31000,64000 AB_NEEDLES=131072 setsid nohup bash bench/speed/abw.sh \
  "@XE_BIN=$B LD_LIBRARY_PATH=$B bash bench/speed/tlppl-ab.sh fsa STRATA_ATTN_FSA=1" \
  fsa0:prompts,needles:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_ATTN_FSA=0 \
  fsa1:prompts,needles:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_ATTN_FSA=1 > logs/abwT157.log 2>&1 < /dev/null &
echo "abwT157 launched: logs/abwT157.log"
