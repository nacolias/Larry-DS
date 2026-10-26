#!/bin/bash
# abwT84: where BAR staging's batches spend the ~0.36 s outside the layers (STRATA_BATCH_TRACE milestones + the pipe
# trace) - base (STRATA_XE_BAR_STAGE=0: the service sets 1 now) against BAR, 8K / 16K twice
cd /opt/ai/dsv41
T=STRATA_PIPE_TRACE=1,STRATA_BATCH_TRACE=1
AB_PROMPTS=8000,16000 setsid nohup bash bench/speed/abw.sh \
  q0:prompts:STRATA_XE_BAR_STAGE=0,$T q1:prompts:$T > logs/abwT84.log 2>&1 < /dev/null &
