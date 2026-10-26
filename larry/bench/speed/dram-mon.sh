#!/bin/bash
# dram-mon.sh OUT [SECONDS] - larry: DRAM traffic (reads + writes, both DDR4 channels) every 100 ms from the AMD data
# fabric counters (needs the amd_uncore module: modprobe amd_uncore).  Writes OUT (perf's CSV) and OUT.start (the
# epoch second it began); dram-sum.py OUT prints the GB/s timeline and its bursts.
OUT=$1; S=${2:-600}
date +%s.%N > $OUT.start
exec perf stat -a -I 100 -x, -e amd_df/dram_channel_data_controller_0/,amd_df/dram_channel_data_controller_1/ -o $OUT -- sleep $S
