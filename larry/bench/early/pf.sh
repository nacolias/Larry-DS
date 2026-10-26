#!/bin/bash
# pf.sh - service off; perf profile of the decode main thread (self + children); a graph-cache debug run; service on
cd /opt/ai/dsv41
touch /run/lock/larry-profile-a.hold; systemctl stop larry-ds41; sleep 5
STRATA_DENSE_REQUANT=q6_K ./perfrun2.sh pf4 --ids-file p1.i32 -n 160
perf report -i /tmp/pf4.perf.data --no-children --percentage relative --sort dso,symbol --stdio -g none 2>/dev/null | grep -v "^#" | grep "%" | head -70 | cut -c1-180 > logs/pf4.self.txt
perf report -i /tmp/pf4.perf.data --no-children --percentage relative --sort dso --stdio -g none 2>/dev/null | grep -v "^#" | grep "%" | head -20 > logs/pf4.dso.txt
DS41_GGML_DEBUG=1 GGML_SYCL_GRAPH_CACHE=1 STRATA_DEC_PROF=1 STRATA_DENSE_REQUANT=q6_K ./run.sh gdbg --ids-file p1.i32 -n 64
rm -f /run/lock/larry-profile-a.hold
timeout 300 /opt/ai/profile-a/larry-mode ds41 > /dev/null 2>&1
echo "== dso"; cat logs/pf4.dso.txt; echo "== self"; head -50 logs/pf4.self.txt
echo "== gdbg"; grep -i "record\|eager\|graph\|dec prof" logs/gdbg.err | sort | uniq -c | sort -rn | head -20
systemctl is-active larry-ds41
