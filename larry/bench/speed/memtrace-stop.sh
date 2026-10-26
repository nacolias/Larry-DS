#!/bin/bash
# memtrace-stop.sh - ends what memtrace-start.sh started (SIGINT: bpftrace prints its end-of-run stacks)
[ -e /run/ds41-memtrace.pids ] || { echo "no memtrace running"; exit 0; }
for p in $(cat /run/ds41-memtrace.pids); do kill -INT $p 2>/dev/null; done
sleep 5
for p in $(cat /run/ds41-memtrace.pids); do kill -0 $p 2>/dev/null && kill -TERM $p 2>/dev/null; done
rm -f /run/ds41-memtrace.pids
echo "memtrace stopped $(date +%T)"
