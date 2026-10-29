#!/bin/bash
# launch-rq.sh - optimization #4's A/B (rqab.sh) in a window once ds41_requant has written the RQ file
cd /opt/ai/dsv41
until grep -q "wrote /srv/dsv41/rq/rq-q20-r9600.rq\|write failed\|cannot" logs/requant-q20-r9600.log; do sleep 30; done
grep -q "wrote" logs/requant-q20-r9600.log || { echo "requant failed: nothing launched" > logs/abw-rq.log; exit 1; }
setsid nohup bash bench/speed/abw.sh "@bash bench/loop/rqab.sh rq" > logs/abw-rq.log 2>&1 < /dev/null &
echo "launched: logs/abw-rq.log"
