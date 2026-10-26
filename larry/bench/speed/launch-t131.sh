#!/bin/bash
# launch-t131.sh - abwT131: a 984K needle on the 12:11 service config - the indexer's column slices
# (STRATA_LI_CHUNK_MIB: a [cap, n] f32 over 512 MiB) only happen past ~256K keys, so their nvis form was not run yet
# (Oct 24: 984K in 1,140 s, decode 61.7 tok/s at that depth)
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_NEEDLES=984000 setsid nohup bash bench/speed/abw.sh lc1:needles > logs/abwT131.log 2>&1 < /dev/null &
echo "abwT131 launched: logs/abwT131.log"
