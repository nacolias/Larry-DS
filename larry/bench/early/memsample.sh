#!/bin/bash
# memsample.sh - every second: time, hidden GiB (MemTotal - MemAvailable - AnonPages), available GiB, TTM system pages
# in use (GiB), VRAM used per card (GiB)
while true; do
  h=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^AnonPages:/{p=$2} END {printf "%d %d", (t-a-p)/1048576, a/1048576}' /proc/meminfo)
  g=$(awk '/usage/{printf "%.1f", $2*4/1048576}' /sys/kernel/debug/dri/0000:15:00.0/gtt_mm)
  v=""
  for d in 15 1b 22 26; do v="$v $(awk '/^  usage/{printf "%.1f", $2/1073741824}' /sys/kernel/debug/dri/0000:$d:00.0/tile0/vram_mm)"; done
  echo "$(date +%H:%M:%S) $h gtt $g vram$v"
  sleep 1
done
