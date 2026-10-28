#!/bin/bash
# launch-t163.sh - abwT163 (#3, part 2): xe_dqgemm v2 (weights decoded into local memory VNNI-packed, a sub-group 32
# tokens x 64 rows: half v1's local-memory bytes a multiply) against v1 - bit-for-bit outputs, times (bench/dqb/dqv2.sh)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@bash bench/dqb/dqv2.sh run" > logs/abwT163.log 2>&1 < /dev/null &
echo "abwT163 launched: logs/abwT163.log"
