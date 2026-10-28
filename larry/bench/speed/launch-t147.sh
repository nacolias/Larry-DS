#!/bin/bash
# launch-t147.sh - abwT147 (#3): how much consecutive queries' compressed-row selections overlap (bench/fa/dsel.sh on the
# staged dselbin: STRATA_DUMP_SEL=1 through p30k and pread)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/fa/dsel.sh" > logs/abwT147.log 2>&1 < /dev/null &
echo "abwT147 launched: logs/abwT147.log"
