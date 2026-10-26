#!/bin/bash
# abwT79: bench/dqb/bar_probe - the CPU writing into card 1's VRAM through its BAR (xe uAPI buffer object, WC mmap):
# GB/s alone / with pinned DMAs, the PRIME export's host-RAM cost, the Level Zero import and a read-back check
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@source /opt/ai/dsv41/xe-env.sh > /dev/null 2>&1; bench/dqb/bar_probe 1024" \
  > logs/abwT79.log 2>&1 < /dev/null &
