#!/bin/bash
# dev-hold.sh on|status - free all four B70s for DeepSeek-V4.1 engine development, the way `larry-mode` does before an
# all-cards layout: Profile A marker (swap.sh / h3 refuse GPU work), swap-guard paused, the studio router + watchdog
# stopped and its lane containers removed, vllm-a stopped.  `larry-mode small` (or a larry-mode all-cards layout) puts
# the box back.
set -u
case "${1:-status}" in
  on)
    touch /run/lock/larry-profile-a
    systemctl stop swap-guard.timer
    systemctl stop studio-watchdog.service studio-router.service
    for c in $(docker ps --format '{{.Names}}' | grep '^studio-'); do docker rm -f "$c" >/dev/null; done
    (cd /opt/docker/vllm && docker compose stop -t 60 vllm-a >/dev/null 2>&1)
    sleep 5 ;;
esac
docker ps --format '{{.Names}} {{.Status}}'
for d in 0 1 2 3; do printf 'xpu%d: ' $d; timeout 10 xpu-smi stats -d $d 2>/dev/null | grep -m1 "GPU Memory Used" | awk -F'|' '{print $3}'; done
