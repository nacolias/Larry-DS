#!/bin/bash
# launch-t144.sh - abwT144: the fused hc path again (STRATA_HC_FUSE_MAXN=16384; off since abwT131's 1M turn died in
# DSV41_HC_FA with UR OUT_OF_RESOURCES - most likely the kept batch's VRAM edge that STRATA_SERVE_KEEP_LEARN_DEPTH_GIB
# now covers): fresh 8K / 16K / 31K x2 on two servers each, interleaved, then the 984K needle + its turn with it on
# (VRAM trace)
cd /opt/ai/dsv41
H=STRATA_HC_FUSE_MAXN=16384
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 AB_NEEDLES=984000 setsid nohup bash bench/speed/abw.sh h0a:prompts h1a:prompts:$H \
  h0b:prompts h1b:prompts:$H lcH:needles:$H,STRATA_VRAM_TRACE=1 > logs/abwT144.log 2>&1 < /dev/null &
echo "abwT144 launched: logs/abwT144.log"
