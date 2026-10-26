#!/bin/bash
# run-probe.sh OUT [li_probe args...] - li_probe and li_probe_cm on card 1 with the clocks pinned (the service must be
# stopped: run.sh's hold + systemctl stop).  Env LI_QW / LI_KR / LI_B / LI_REPS pass through.
OUT=$1; shift
set --  "$@"
ARGS=("$@")
set --
source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
cd /opt/ai/dsv41/bench/li
if systemctl is-active --quiet larry-ds41 || pgrep -x ds41_generate_x > /dev/null; then echo "the service / an engine is running" > $OUT; exit 3; fi
/usr/local/sbin/larry-gpu-clock pin > /dev/null
{ for P in ${PROBES:-li_probe li_probe_cm}; do echo "== $P $(date +%T)"; ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 600 ./$P "${ARGS[@]}"; echo "rc=$?"; done; } > $OUT 2>&1
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo "CAT errors since boot: $(dmesg | grep -c -i 'cat error')" >> $OUT
