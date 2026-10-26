#!/bin/bash
# launch-t102.sh - the A/B window once the user is done testing (AB_IDLE_MIN=0: at once if the service is idle).
#   1. tlppl-shexp.sh: STRATA_SHEXP_TIER=1's numerics (p2k / p8k PPL, KL against the run-to-run floor)
#   2. the agent replay (agentx.py: a coding session growing to ~43K, follow-ups of 17-6,500 tokens) without and with
#      STRATA_SERVE_KEEP_LEARN_HIST=1 + STRATA_SERVE_REGROW_ASYNC=1, twice each (interleaved)
#   3. fresh 8K / 16K / 31K prompts without and with STRATA_SHEXP_TIER=1
#   4. the agent replay traced (STRATA_XE_CALL_TL, STRATA_BATCH_TRACE, STRATA_PIPE_TRACE): where a follow-up's tier
#      calls spend their time
cd /opt/ai/dsv41
NEW=STRATA_SERVE_KEEP_LEARN_HIST=1,STRATA_SERVE_REGROW_ASYNC=1
AB_IDLE_MIN=0 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/smoke102.sh" \
  "@bash bench/speed/tlppl-shexp.sh" \
  k0:agent k1:agent:$NEW k0b:agent k1b:agent:$NEW \
  s0:prompts s1:prompts:STRATA_SHEXP_TIER=1 \
  t1:agent:$NEW,STRATA_XE_CALL_TL=1,STRATA_BATCH_TRACE=1,STRATA_PIPE_TRACE=1 \
  > logs/abwT102.log 2>&1 < /dev/null &
echo "abwT102 launched: logs/abwT102.log"
