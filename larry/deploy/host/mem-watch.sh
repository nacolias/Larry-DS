#!/bin/bash
# mem-watch.sh - memory watchdog for the ds41 server (ds41-watch.service): the xe driver backs GPU buffers with host
# RAM that swap cannot reclaim and an OOM daemon cannot see.  If that "hidden" RAM (MemTotal - MemAvailable -
# AnonPages) passes WATCH_HIDDEN_LIMIT_GIB, or MemAvailable drops below WATCH_AVAIL_FLOOR_GIB, the server's main
# process is killed before the kernel deadlocks (larry hung on 2026-09-24, and rebooted on 2026-10-19 when a pinning
# probe beside an engine took the driver's host memory past TTM's limit: ~60.7 GiB of its 121).  The healthy service
# on larry peaks at ~33 GiB hidden.
#   WATCH_UNIT (ds41.service)  WATCH_HIDDEN_LIMIT_GIB (54)  WATCH_AVAIL_FLOOR_GIB (6)  WATCH_LOG (/var/log/ds41-watch.log)
set -u
LIMIT=${WATCH_HIDDEN_LIMIT_GIB:-54}
FLOOR=${WATCH_AVAIL_FLOOR_GIB:-6}
LOG=${WATCH_LOG:-/var/log/ds41-watch.log}
UNIT=${WATCH_UNIT:-ds41.service}

while sleep 2; do
  pid=$(systemctl show -p MainPID --value "$UNIT" 2>/dev/null)
  case "$pid" in ''|0|*[!0-9]*) continue ;; esac
  read -r hidden avail < <(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^AnonPages:/{p=$2}
                                END{printf "%d %d\n", (t-a-p)/1048576, a/1048576}' /proc/meminfo)
  if [ "$hidden" -gt "$LIMIT" ] || [ "$avail" -lt "$FLOOR" ]; then
    msg="hidden=${hidden}GiB (limit ${LIMIT}) avail=${avail}GiB (floor ${FLOOR}) -> killing $UNIT main pid $pid"
    echo "$(date -Is) $msg" >>"$LOG"
    logger -t ds41-watch "$msg"
    kill -9 "$pid"
    exit 0
  fi
done
