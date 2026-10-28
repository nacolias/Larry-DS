#!/bin/bash
# launch-t164.sh - abwT164 (#3, part 2): xe_dqgemm v2 in the engine (staged dq2bin = the service + patch-dqv2.py):
# STRATA_XE_DQ2 off / on, fresh 16K / 31K / 64K prompts and a 131K needle (same binary; the outputs are the same bits)
cd /opt/ai/dsv41
B=/opt/ai/dsv41/dq2bin
AB_PROMPTS=16000,31000,64000 AB_NEEDLES=131072 setsid nohup bash bench/speed/abw.sh \
  dq0:prompts,needles:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_XE_DQ2=0 \
  dq1:prompts,needles:XE_BIN=$B,LD_LIBRARY_PATH=$B,STRATA_XE_DQ2=1 > logs/abwT164.log 2>&1 < /dev/null &
echo "abwT164 launched: logs/abwT164.log"
