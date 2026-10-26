#!/bin/bash
# abwT80: STRATA_XE_BAR_STAGE=1 (patch-barstage.py: pageable experts written into the cards' VRAM through their BARs)
# - bc: its correctness (every staged / streamed expert read back and compared, prof lines), b0: base, b1: BAR staging;
# fresh prompts 512 .. 31K twice each
cd /opt/ai/dsv41
AB_PROMPTS=512,2000,4500,8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  bc:prompts:STRATA_XE_BAR_STAGE=1,STRATA_XE_HOTRING_CHECK=1,STRATA_XE_PROF=1 b0:prompts b1:prompts:STRATA_XE_BAR_STAGE=1 \
  > logs/abwT80.log 2>&1 < /dev/null &
