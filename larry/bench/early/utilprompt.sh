#!/bin/bash
# utilprompt.sh NAME APPROX_TOKENS - sample all four cards with xpu-smi while the ds41 service takes a cold
# needle prompt of ~N tokens; prints each card's averages over the prompt's window (samples with power above idle)
N=${1:-utilp}; T=${2:-31000}
D=/opt/ai/dsv41/logs
F="utilization.compute,memory.read.bandwidth,memory.bandwidth.utilization,eu.active,eu.stall,pcie.rx.throughput.kbs,clocks.current.graphics,power.draw"
PIDS=""
for d in 0 1 2 3; do timeout 200 xpu-smi dump -d $d --metrics $F --loop-ms 500 --time 120 > $D/$N.card$d.csv 2>&1 & PIDS="$PIDS $!"; done
sleep 3
python3 /opt/ai/dsv41/needle.py $T 1 | tail -2 | head -1
sleep 2; kill $PIDS 2>/dev/null; sleep 1
python3 - "$D/$N" <<'PY'
import sys, csv, statistics as st
base = sys.argv[1]
names = ["compute%", "memrd MB/s", "membw%", "eu.active%", "eu.stall%", "pcie.rx KB/s", "gfx MHz", "power W"]
for d in range(4):
    rows = [r for r in csv.reader(open(f"{base}.card{d}.csv")) if r and r[0][:1].isdigit()]
    if not rows: print("card", d, "no data"); continue
    def f(x):
        try: return float(x)
        except ValueError: return None
    busy = [r for r in rows if (f(r[9]) or 0) > 60]   # power above idle: the prompt's window
    if not busy: print("card", d, "never busy"); continue
    out = []
    for i, n in enumerate(names):
        v = [f(r[i + 2]) for r in busy if f(r[i + 2]) is not None]
        out.append(f"{n} {st.mean(v):.0f}" if v else f"{n} -")
    print(f"card {d} ({len(busy)} busy samples): " + ", ".join(out))
PY
