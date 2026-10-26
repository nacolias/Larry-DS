#!/bin/bash
# launch-t113.sh - abwT113: fresh 16K / 31K prompts on the service's config (the fused prompt indexer on), traced
# (STRATA_PIPE_TRACE: each layer's dense / join / finish steps; STRATA_XE_CALL_TL: each tier call per card) - with card 0's
# indexer ~5 ms a call lighter, which side bounds a big prompt's layers 0-19 now? (abwT104 before: D 3.9-4.3 s, J 0.3 s)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=16000,31000 setsid nohup bash bench/speed/abw.sh \
  tr1:prompts:STRATA_PIPE_TRACE=1,STRATA_XE_CALL_TL=1 \
  > logs/abwT113.log 2>&1 < /dev/null &
echo "abwT113 launched: logs/abwT113.log"
