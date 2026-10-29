#!/bin/bash
# launch-alinst.sh - the answer guard into the service: patch-answerloop.py (the server; its tests in
# serve/test_reasoning_loop_recovery.py) + svc-answerloop.py ("answer_loop_stop": 0.9, backup
# larry-ds41.json.pre-answerloop-20261029) in a window (its restart loads them); then post-lw.sh alinst: tsvc2 +
# concurtest + both loop probes (logs/tsvc2-alinst.log)
cd /opt/ai/dsv41
setsid nohup bash bench/speed/abw.sh "@python3 patches/patch-answerloop.py && python3 patches/svc-answerloop.py" > logs/abw-alinst.log 2>&1 < /dev/null &
setsid nohup bash bench/loop/post-lw.sh alinst > /dev/null 2>&1 < /dev/null &
echo "launched: logs/abw-alinst.log, then logs/tsvc2-alinst.log"
