#!/bin/bash
# nicebuild.sh [TARGETS] - build-xe.sh at the lowest CPU / IO priority while the service runs, frozen (SIGSTOP to the
# build's process group) whenever the service has a request running or queued and thawed when it is idle again, so a
# user's request never shares the CPU with the compiler.  The running engine keeps its own (old) binaries; the next
# start picks the new ones.  Log: logs/nicebuild.log.
set -u
cd /opt/ai/dsv41
KEY=$(grep ^VLLM_API_KEY /opt/docker/vllm/.env | cut -d= -f2-)
busy() {   # 0 = the service has work (or cannot tell), 1 = idle
  curl -s -m 3 -H "Authorization: Bearer $KEY" http://172.23.0.1:18080/metrics 2>/dev/null |
    python3 -c "import json,sys; d=json.load(sys.stdin)['live']; sys.exit(1 if d['state']=='idle' and not d['queued'] else 0)" 2>/dev/null
}
echo "build start $(date '+%F %T') targets ${1:-ds41_generate_xe}"
TARGETS="${1:-ds41_generate_xe}" setsid nice -n 19 ionice -c3 bash /opt/ai/dsv41/build-xe.sh > logs/nicebuild-ninja.log 2>&1 < /dev/null &
BP=$!
sleep 1
PG=$(ps -o pgid= -p $BP | tr -d ' ')
stopped=0; nstop=0
while kill -0 $BP 2>/dev/null; do
  if systemctl is-active -q larry-ds41 && busy; then
    if [ $stopped = 0 ]; then kill -STOP -- -$PG 2>/dev/null; stopped=1; nstop=$((nstop+1)); echo "$(date +%T) request running: build frozen"; fi
  elif [ $stopped = 1 ]; then
    kill -CONT -- -$PG 2>/dev/null; stopped=0; echo "$(date +%T) idle: build resumed"
  fi
  sleep 2
done
[ $stopped = 1 ] && kill -CONT -- -$PG 2>/dev/null
wait $BP; rc=$?
echo "build end $(date '+%F %T') rc=$rc (frozen $nstop times)"
tail -3 logs/nicebuild-ninja.log
