#!/bin/bash
# launch-t166.sh - abwT166: abwT165's two 31K prompts stalled in the expert phase (11.4 / 14.5 s against 9.7 / 10.1 in
# abwT164 on the same config): 16K and 31K again for the front page's table
cd /opt/ai/dsv41
AB_PROMPTS=16099,31008 setsid nohup bash bench/speed/abw.sh now2:prompts > logs/abwT166.log 2>&1 < /dev/null &
echo "abwT166 launched: logs/abwT166.log"
