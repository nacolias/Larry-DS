#!/bin/bash
# engmon.sh OUT [SECONDS] [MS] - larry: each card's copy engine (bcs0) and compute engine (ccs0) busy ticks every MS
# (default 100) from the xe PMUs (perf); engsum.py OUT prints busy % per interval.  OUT.start = the epoch second.
OUT=$1; S=${2:-600}; I=${3:-100}
E=""
for P in xe_0000_15_00.0 xe_0000_1b_00.0 xe_0000_22_00.0 xe_0000_26_00.0; do
  E="$E,$P/engine-active-ticks,engine_class=1,engine_instance=0,gt=0/,$P/engine-total-ticks,engine_class=1,engine_instance=0,gt=0/"
  E="$E,$P/engine-active-ticks,engine_class=4,engine_instance=0,gt=0/"
done
date +%s.%N > $OUT.start
exec perf stat -a -I $I -x ';' -e ${E#,} -o $OUT -- sleep $S
