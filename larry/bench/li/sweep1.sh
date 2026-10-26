#!/bin/bash
# sweep1.sh - li_probe_cm over work-group shapes (KR keys x QW queries a work-group) at the service's span shapes
cd /opt/ai/dsv41/bench/li
for KR in 256 512 1024 2048; do for QW in 4 8 16; do
  echo "== KR $KR QW $QW"
  LI_KR=$KR LI_QW=$QW LI_REPS=20 PROBES=li_probe_cm ./run-probe.sh sw.tmp 4096:3900 16384:8400 16384:15800 32768:31600 65536:60000 262144:250000
  grep -E "^ +[0-9]+ +[0-9]+ \|" sw.tmp
done; done
grep "CAT" sw.tmp
