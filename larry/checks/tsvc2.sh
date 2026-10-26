#!/bin/bash
# tsvc2.sh - the service check: a short needle, decode (svcbench, essay), fresh needles at 8K / 8K / 16K / 31K tokens
# (prompt tok/s, FOUND), decode again, a tool call - with the memory the GPU driver holds sampled every 2 s (hidden =
# MemTotal - MemAvailable - AnonPages, GiB; larry's watchdog kills the server at 54).  The server: ds41api.py
# (DS41_URL, DS41_API_KEY / DS41_KEY_FILE); DS41_UNIT (ds41; larry: larry-ds41).
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$D" && mkdir -p logs
( while true; do awk "/^MemTotal:/{t=\$2} /^MemAvailable:/{a=\$2} /^AnonPages:/{p=\$2} END {printf \"%s %d %d\n\", strftime(\"%H:%M:%S\"), (t-a-p)/1048576, a/1048576}" /proc/meminfo; sleep 2; done ) > logs/hidden.log 2>&1 &
MP=$!
python3 needle.py 200 1 | tail -1
python3 svcbench.py 2>&1 | tail -1
python3 essay.py 3 2>&1 | tail -1
for n in 8000 8000 16000 31000; do python3 needle.py $n 1 2>&1 | tail -2 | tr "\n" " " | grep -o "prompt_per_second.: [0-9.]*\|FOUND\|MISSED" | tr "\n" " "; echo; done
python3 svcbench.py 2>&1 | tail -1
python3 essay.py 3 2>&1 | tail -1
python3 tooltest.py 2>&1 | tail -2 | head -1 | cut -c1-80
kill $MP
echo "hidden GiB max $(awk "{print \$2}" logs/hidden.log | sort -n | tail -1), available GiB min $(awk "{print \$3}" logs/hidden.log | sort -n | head -1)"
grep -h "seeded with\|promotion cache:" logs/engine.log | tail -3 | cut -c1-160
systemctl is-active "${DS41_UNIT:-ds41}"
