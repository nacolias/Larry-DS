#!/bin/bash
# probe-span4.sh - vmm_span_probe4 (built here if missing) as an abw.sh "@" step: how card 0.s VMM reservation must grow for the peers to read it right
# at the engine's sizes and order.  A page fault: every card gets a GT reset before the window brings the service back.
cd /opt/ai/dsv41/bench/conc
set -- ; source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
[ -x vmm_span_probe4 ] || icpx -fsycl -O2 vmm_span_probe4.cpp -lze_loader -o vmm_span_probe4 || { echo "build failed"; exit 1; }
T0=$(date '+%F %T')
./vmm_span_probe4; echo "rc=$?"
f=$(journalctl -k --since "$T0" --no-pager | grep -c "Fault response")
echo "page faults in the kernel log since the probe started: $f"
if [ "$f" -gt 0 ]; then
  journalctl -k --since "$T0" --no-pager | grep -A 8 "Faulted Address" | head -12
  bash /opt/ai/dsv41/logs/gtreset.sh
fi
