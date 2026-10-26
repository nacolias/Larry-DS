#!/bin/bash
# abw.sh NAME:TESTS[:VAR=v,VAR=v] ... - the 2026-10-24 afternoon A/B window.  Waits until the service has been idle for
# AB_IDLE_MIN minutes (default 20: no request finished since, none running or queued) - at most AB_WAIT_H hours (6) -
# then holds and stops it, stands in on :18080 (yieldsrv.py: any request gets a 503 and raises the flag), runs tabx.sh
# per config in order (stopping at the flag), and brings the service back.  New CAT errors in dmesg after a config:
# stop with the hold kept (the cards need a GT reset first).
#   abw.sh base:prompts,edit,dec dl:prompts,edit,dec:STRATA_XE_PREFETCH_DEPTH=8192,STRATA_LOOKUP_TP=1
# A config "@COMMAND" runs COMMAND (a probe: cards to itself, clocks pinned) in the window instead.
set -u
cd /opt/ai/dsv41
S=bench/speed
FLAG=/run/ds41-ab.yield
export AB_FLAG=$FLAG
PY=/opt/ai/dsv41/venv/bin/python
KEY=$(grep ^VLLM_API_KEY /opt/docker/vllm/.env | cut -d= -f2-)
cat_errs() { dmesg | grep -c -i "cat error"; }
# CAT errors logged since the window started (the kernel journal: dmesg's ring buffer drops old lines)
cat_new() { journalctl -k --since "$WSTART" --no-pager 2>/dev/null | grep -c -i "cat error"; }
idle_min() {   # minutes since the service's last finished request (999: none in 12 h)
  local t
  t=$(journalctl -u larry-ds41 --since "-12h" --no-pager -o short-unix 2>/dev/null | grep "done:" | tail -1 | cut -d' ' -f1 | cut -d. -f1)
  [ -z "$t" ] && { echo 999; return; }
  echo $(( ($(date +%s) - t) / 60 ))
}
live_idle() {   # the server says idle and nothing queued
  curl -s -m 5 -H "Authorization: Bearer $KEY" http://172.23.0.1:18080/metrics 2>/dev/null |
    python3 -c "import json,sys; d=json.load(sys.stdin)['live']; sys.exit(0 if d['state']=='idle' and not d['queued'] else 1)" 2>/dev/null
}
restore() {
  [ -n "${YP:-}" ] && kill $YP 2>/dev/null
  for i in $(seq 1 20); do ss -ltn | grep -q ":18080 " || break; sleep 1; done
  rm -f /run/lock/larry-profile-a.hold
  larry-mode ds41
  local c=0
  for i in $(seq 1 60); do
    c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://172.23.0.1:18080/health)
    [ "$c" = "200" ] && break
    sleep 5
  done
  echo "service health $c $(date '+%F %T')"
}
deadline=$(( $(date +%s) + ${AB_WAIT_H:-6} * 3600 ))
while :; do
  m=$(idle_min)
  if [ "$m" -ge "${AB_IDLE_MIN:-20}" ] && live_idle && systemctl is-active -q larry-ds41 && [ ! -e /run/lock/larry-profile-a.hold ]; then break; fi
  [ "$(date +%s)" -gt $deadline ] && { echo "not idle for ${AB_IDLE_MIN:-20} min within ${AB_WAIT_H:-6} h - nothing run"; exit 2; }
  echo "$(date +%T) waiting: last request $m min ago"
  sleep 60
done
echo "window start $(date '+%F %T') (last request $(idle_min) min ago), CAT errors in dmesg: $(cat_errs)"
c0=$(cat_errs)
WSTART=$(date '+%F %T')
touch /run/lock/larry-profile-a.hold
systemctl stop larry-ds41
for i in $(seq 1 60); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running after the stop: aborting, hold kept"; exit 1; fi
for i in $(seq 1 20); do ss -ltn | grep -q ":18080 " || break; sleep 1; done
rm -f $FLAG
setsid nohup $PY -I $S/yieldsrv.py $FLAG --port 18080 --max-s 5400 > logs/yieldsrv.log 2>&1 < /dev/null &
YP=$!
echo "service stopped $(date +%T); stand-in pid $YP"
for CFG in "$@"; do
  [ -e $FLAG ] && { echo "a request came in ($(cat $FLAG)) - stopping the A/B"; break; }
  if [ "${CFG:0:1}" = "@" ]; then   # @COMMAND: a probe run as it is (clocks pinned), no server
    echo "probe: ${CFG:1}  $(date +%T)"
    /usr/local/sbin/larry-gpu-clock pin > /dev/null
    bash -c "${CFG:1}" 2>&1 | sed -u "s/^/[probe] /"
    /usr/local/sbin/larry-gpu-clock restore > /dev/null
    c=$(cat_new)
    if [ "$c" -gt 0 ]; then
      echo "NEW CAT ERRORS in the kernel log ($c since $WSTART) after the probe: stopping, service NOT restarted, hold kept"
      kill $YP 2>/dev/null
      exit 1
    fi
    continue
  fi
  N=${CFG%%:*}; R=${CFG#*:}
  T=${R%%:*}; E=${R#*:}
  [ "$E" = "$R" ] && E=
  bash $S/tabx.sh $N $T ${E//,/ }
  c=$(cat_new)
  if [ "$c" -gt 0 ]; then
    echo "NEW CAT ERRORS in the kernel log ($c since $WSTART) after $N: stopping, service NOT restarted, hold kept"
    kill $YP 2>/dev/null
    exit 1
  fi
done
for i in $(seq 1 90); do pgrep -x ds41_generate_x > /dev/null || break; sleep 2; done
if pgrep -x ds41_generate_x > /dev/null; then echo "an engine is still running: service NOT restarted, hold kept"; kill $YP 2>/dev/null; exit 1; fi
restore
[ -e $FLAG ] && echo "the window was cut short by a request: $(cat $FLAG)"
echo "window end $(date '+%F %T'), CAT errors in dmesg: $(cat_errs)"
