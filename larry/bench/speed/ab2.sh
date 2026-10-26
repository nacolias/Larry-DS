#!/bin/bash
# ab2.sh - round 2 of the CPU-light A/B: pp2048 at depth 0 (the prefetch's range), pp96 (the DSpark tail's passes vs a
# batch), pp512 at depth 32768; T = 0 / 4 / 4 with the prefetch from 4096 tokens and batches from 64; then a one-shot
# T = 4 run of p512 (the close() fix: rc must be 0).  The service held and stopped, then back (tab_all.sh).
cd /opt/ai/dsv41
export TAB_SPECS="2048:0 96:0 512:32768"
TAB_ONESHOT=1 TAB_REPS=5 bash bench/speed/tab_all.sh r0:STRATA_XE_CPU_LIGHT=0 r4:STRATA_XE_CPU_LIGHT=4 \
  r4x:STRATA_XE_CPU_LIGHT=4,STRATA_XE_PREFETCH_MIN=4096,STRATA_SERVE_BATCH_MIN=64
echo "round 2 done $(date +%T)"
