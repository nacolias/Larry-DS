#!/bin/bash
# launch-t161.sh - abwT161 (#3, part 2): where an expert card's prompt call spends its GPU time now (the service's engine
# with the fused attention + FN16=2 under unitrace: utdrive.py traces one fresh 16K prompt after a 16K warm-up; then
# bench/speed/utcall.py per card and engine)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@/opt/ai/dsv41/venv/bin/python -I bench/speed/utdrive.py utfsa 16000 --warm 16000" \
  > logs/abwT161.log 2>&1 < /dev/null &
echo "abwT161 launched: logs/abwT161.log"
