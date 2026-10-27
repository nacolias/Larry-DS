#!/bin/bash
# probe-span.sh - vmm_span_probe (built here if missing) as an abw.sh "@" step: F and G first, E last (E may fault a card:
# then every card gets a GT reset and copy_probe here, before the window brings the service back)
cd /opt/ai/dsv41/bench/conc
set -- ; source /opt/intel/oneapi/2026.1/oneapi-vars.sh > /dev/null 2>&1
[ -x vmm_span_probe ] || icpx -fsycl -O2 vmm_span_probe.cpp -lze_loader -o vmm_span_probe || { echo "build failed"; exit 1; }
T0=$(date '+%F %T')
./vmm_span_probe FG; echo "FG rc=$?"
./vmm_span_probe FGE; echo "FGE rc=$?"
f=$(journalctl -k --since "$T0" --no-pager | grep -c "Fault response")
echo "page faults in the kernel log since the probe started: $f"
if [ "$f" -gt 0 ]; then
  journalctl -k --since "$T0" --no-pager | grep -A 8 "Faulted Address" | head -12
  bash /opt/ai/dsv41/logs/gtreset.sh
fi
