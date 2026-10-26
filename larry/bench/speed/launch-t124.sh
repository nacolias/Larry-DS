#!/bin/bash
# launch-t124.sh - abwT124: the fused hc ops at prompt widths (STRATA_HC_FUSE_MAXN=16384: hc_fa / hc_fb with
# patch-hcwide.py's wide hc_fa, the outputs written in place - STRATA_DIRECT_OUT) on top of the service's config
# (OPROJ_A, HC_NORM_MV_WIDE, LI_FUSED): 1. numerics  2. card 0's op profile, by shape  3. fresh prompts  4. agent replay
cd /opt/ai/dsv41
cp -n logs/opps_p30k.err logs/opps_p30k-T122.err 2>/dev/null
B=STRATA_HC_FUSE_MAXN=16384
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-ab.sh hcw $B" \
  "@SKIP=42000 STRATA_HC_FUSE_MAXN=16384 bash bench/speed/opprof-shapes.sh p30k.i32" \
  v0:prompts v1:prompts:$B k0:agent k1:agent:$B \
  > logs/abwT124.log 2>&1 < /dev/null &
echo "abwT124 launched: logs/abwT124.log"
