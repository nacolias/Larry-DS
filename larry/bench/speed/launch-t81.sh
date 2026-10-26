#!/bin/bash
# abwT81: bench/dqb/bar_leak - does bar_free give a BAR buffer's VRAM back (4 variants of the free)?
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1; bench/dqb/bar_leak 0 1 2 3" \
  > logs/abwT81.log 2>&1 < /dev/null &
