#!/bin/bash
# abwT90: STRATA_PIPE_MIN_PART (patch-pipesplit.py) - a 530-token batch split 512 + 18 (an 18-token tier call a layer,
# ~12-16 ms each); 0 (as before) / 256 / 512, and no pipelining at all; fresh 512 / 700 / 1K / 2K / 3K twice each
cd /opt/ai/dsv41
AB_PROMPTS=512,700,1000,2000,3000 setsid nohup bash bench/speed/abw.sh \
  m0:prompts m256:prompts:STRATA_PIPE_MIN_PART=256 m512:prompts:STRATA_PIPE_MIN_PART=512 np:prompts:STRATA_PIPE=0 \
  > logs/abwT90.log 2>&1 < /dev/null &
