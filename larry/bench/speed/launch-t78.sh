#!/bin/bash
# abwT78: the LM head split through card-0 staging (patch-headstage.py) - hc: its exactness self-check against the
# direct copies (STRATA_XE_HEAD_STAGE_CHECK=1), h0: the direct copies (STRATA_XE_HEAD_STAGE=0), h1: staged (default);
# prompts 8K / 16K / 31K twice + the decode test, under xetrace.sh (do the per-prompt dma-buf exports go?)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/xetrace.sh logs/abwT78.log > logs/xetrace78.log 2>&1 < /dev/null &
G=STRATA_PIPE_TRACE=1
AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  hc:prompts:STRATA_XE_HEAD_STAGE_CHECK=1,$G h0:prompts,dec:STRATA_XE_HEAD_STAGE=0,$G h1:prompts,dec:$G \
  > logs/abwT78.log 2>&1 < /dev/null &
