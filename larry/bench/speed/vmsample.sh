#!/bin/bash
# vmsample.sh OUT [SECONDS] - larry: every 0.1 s the epoch ms and /proc/vmstat's reclaim / compaction / swap counters
# (allocstall_normal allocstall_movable compact_stall pgmajfault pswpin pswpout) - a prompt stall that lines up with
# direct reclaim or compaction shows as a jump.  Stops after SECONDS (default 1800).
OUT=$1; END=$(( $(date +%s) + ${2:-1800} ))
while [ "$(date +%s)" -lt $END ]; do
  echo "$(date +%s%3N) $(awk '/^(allocstall_normal|allocstall_movable|compact_stall|pgmajfault|pswpin|pswpout) /{printf "%s ", $2}' /proc/vmstat)" >> $OUT
  sleep 0.1
done
