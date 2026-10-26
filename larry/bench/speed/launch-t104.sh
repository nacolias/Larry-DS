#!/bin/bash
# launch-t104.sh - abwT104: fresh 16K / 31K prompts on the service's config, traced (STRATA_PIPE_TRACE: each layer's
# dense / join / finish steps; STRATA_XE_CALL_TL: each tier call per card) - after the shared expert moved to the
# expert cards, which side bounds a big prompt's layers now?
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=16000,31000 setsid nohup bash bench/speed/abw.sh \
  tr0:prompts:STRATA_PIPE_TRACE=1,STRATA_XE_CALL_TL=1 \
  > logs/abwT104.log 2>&1 < /dev/null &
echo "abwT104 launched: logs/abwT104.log"
