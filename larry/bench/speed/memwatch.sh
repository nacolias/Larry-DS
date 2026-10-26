#!/bin/bash
# memwatch.sh OUT [SECONDS] [HIDDEN_MAX_GIB] [AVAIL_MIN_GIB] - larry: every 2 s MemAvailable, AnonPages and "hidden"
# (MemTotal - MemAvailable - AnonPages: pinned / driver memory) in GiB; when hidden passes HIDDEN_MAX (50) or
# MemAvailable drops under AVAIL_MIN (8) - an A/B engine on :18081 has no watchdog - it raises the A/B stop flag and
# SIGTERMs the engine (a clean exit; never -9 with copies in flight).  Stops after SECONDS (1800).
OUT=$1; END=$(( $(date +%s) + ${2:-1800} )); HMAX=${3:-50}; AMIN=${4:-8}
while [ "$(date +%s)" -lt $END ]; do
  read -r tot avail anon < <(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^AnonPages:/{n=$2} END{print t, a, n}' /proc/meminfo)
  h=$(( (tot - avail - anon) / 1048576 )); a=$(( avail / 1048576 )); n=$(( anon / 1048576 ))
  echo "$(date +%T) avail $a anon $n hidden $h" >> $OUT
  if [ "$h" -gt "$HMAX" ] || [ "$a" -lt "$AMIN" ]; then
    echo "$(date +%T) LIMIT (hidden $h > $HMAX or avail $a < $AMIN): stopping the A/B engine" >> $OUT
    touch /run/ds41-ab.yield
    p=$(pgrep -x ds41_generate_x); [ -n "$p" ] && kill -TERM $p
    break
  fi
  sleep 2
done
