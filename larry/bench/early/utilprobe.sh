#!/bin/bash
# utilprobe.sh NAME MAXTOK - sample all four cards with xpu-smi while the ds41 service writes MAXTOK tokens
N=${1:-util}; T=${2:-800}
D=/opt/ai/dsv41/logs
F="utilization.compute,memory.read.bandwidth,memory.bandwidth.utilization,eu.active,eu.stall,pcie.rx.throughput.kbs,clocks.current.graphics,power.draw"
for d in 0 1 2 3; do (timeout 300 xpu-smi dump -d $d --metrics $F --loop-ms 500 --time 240 > $D/$N.card$d.csv 2>&1 &) ; done
sleep 3
KEY=$(grep -E '^VLLM_API_KEY' /opt/docker/vllm/.env | cut -d= -f2)
curl -s http://172.23.0.1:18080/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"larry\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed, long technical essay (at least 1500 words) about how virtual memory works in modern operating systems.\"}],\"max_tokens\":$T,\"reasoning_effort\":\"none\"}" > $D/$N.json
python3 -c "import json; d=json.load(open('$D/$N.json')); print('timings', d['timings'])"
sleep 2; pkill -f "xpu-smi dump" ; sleep 1
python3 - "$D/$N" <<'PY'
import sys, csv, statistics as st
base = sys.argv[1]
for d in range(4):
    rows = [r for r in csv.reader(open(f"{base}.card{d}.csv")) if r and r[0][:1].isdigit()]
    if not rows: print("card", d, "no data"); continue
    # columns: Timestamp, DeviceId, then metrics in order
    cols = list(zip(*rows))
    def m(i):
        v = [float(x) for x in cols[i] if x.strip() not in ("N/A", "")]
        return (st.mean(v), max(v)) if v else (float("nan"), float("nan"))
    names = ["compute%", "memrd MB/s", "membw%", "eu.active%", "eu.stall%", "pcie.rx KB/s", "gfx MHz", "power W"]
    # keep only the busy middle: samples where power is above idle
    print(f"card {d}: " + ", ".join(f"{n} {m(i+2)[0]:.0f} (max {m(i+2)[1]:.0f})" for i, n in enumerate(names)))
PY
