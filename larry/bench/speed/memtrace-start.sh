#!/bin/bash
# memtrace-start.sh TAG [MINUTES] - memtrace.bt (or BT=memtrace2.bt; bpftrace) and vmwatch2.py in the background for at most MINUTES (60):
# logs/memtrace-TAG.log, logs/vmwatch2-TAG.jsonl; their PIDs in /run/ds41-memtrace.pids for memtrace-stop.sh
cd /opt/ai/dsv41
T=$1; M=${2:-60}
S=bench/speed
setsid nohup timeout -s INT $((M * 60)) bpftrace -B line $S/${BT:-memtrace.bt} > logs/memtrace-$T.log 2>&1 < /dev/null &
echo $! > /run/ds41-memtrace.pids
setsid nohup timeout $((M * 60)) python3 -I $S/vmwatch2.py logs/vmwatch2-$T.jsonl $((M * 60)) > /dev/null 2>&1 < /dev/null &
echo $! >> /run/ds41-memtrace.pids
sleep 8
echo "memtrace $T started $(date +%T), pids $(tr '\n' ' ' < /run/ds41-memtrace.pids)"
head -3 logs/memtrace-$T.log
