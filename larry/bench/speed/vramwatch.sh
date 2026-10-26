#!/bin/bash
# vramwatch.sh OUT [SECONDS] - every 0.2 s: epoch ms, per card free VRAM (MiB, xe debugfs vram_mm) and GTT usage (MiB:
# buffers in system memory - a jump = VRAM spilled), MemAvailable (MiB).  Stops after SECONDS (default 1800).
OUT=$1; END=$(( $(date +%s) + ${2:-1800} ))
D="0000:15:00.0 0000:1b:00.0 0000:22:00.0 0000:26:00.0"
while [ "$(date +%s)" -lt $END ]; do
  l="$(date +%s%3N)"
  for d in $D; do
    f=$(sed -n "s/^chunk_size.*, free: \([0-9]*\)MiB,.*/\1/p" /sys/kernel/debug/dri/$d/tile0/vram_mm 2>/dev/null)
    g=$(awk '/usage/{printf "%d", $2/1048576}' /sys/kernel/debug/dri/$d/gtt_mm 2>/dev/null)
    l="$l $f $g"
  done
  l="$l $(awk '/^MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)"
  echo "$l" >> $OUT
  sleep 0.2
done
