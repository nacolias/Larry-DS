#!/bin/bash
# abwT87: short fresh prompts with BAR staging (the service's) - the CPU pool's share: STRATA_XE_CPU_LIGHT 4 (the
# service) / 3 / 2, and 7 pool workers; 512 / 1K / 2K / 3K twice each
cd /opt/ai/dsv41
AB_PROMPTS=512,1000,2000,3000 setsid nohup bash bench/speed/abw.sh \
  l4:prompts l3:prompts:STRATA_XE_CPU_LIGHT=3 l2:prompts:STRATA_XE_CPU_LIGHT=2 p7:prompts:STRATA_XE_POOL_WORKERS=7 \
  > logs/abwT87.log 2>&1 < /dev/null &
