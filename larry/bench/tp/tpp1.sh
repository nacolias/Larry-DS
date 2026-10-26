#!/bin/bash
# tpp1.sh - the 31.6K prompt (p30k.i32, --prompt-batch 16384, service settings) with xpu-smi on all four cards
# (250 ms) and the engine's own phase profiles: where card 0's idle time is
cd /opt/ai/dsv41
source xe-env.sh
/usr/local/sbin/larry-gpu-clock pin > /dev/null
N=${1:-pp31a}
B="STRATA_XE_PINNED_GIB=24 STRATA_XE_DENSE_GIB=14 STRATA_XE_DENSE_RESERVE_GIB=4 STRATA_XE_RESERVE_GIB=3.5 STRATA_ENGRAM_REQUANT=q6_K STRATA_DENSE_REQUANT=q6_K STRATA_D3_PF=0 STRATA_DENSE_PROF=1 STRATA_ENG_PROF=1 STRATA_XE_PROF=1"
F="utilization.compute,memory.read.bandwidth,memory.write.bandwidth,eu.active,power.draw,pcie.rx.throughput.kbs"
rm -f logs/$N.err logs/$N.marks
PIDS=""
for d in 0 1 2 3; do xpu-smi dump -d $d --metrics $F --loop-ms 250 > logs/$N.card$d.csv 2>&1 & PIDS="$PIDS $!"; done
env $B RUN_TIMEOUT=1500 ./run.sh $N --ids-file p30k.i32 -n 8 --prompt-batch 16384 > /dev/null 2>&1 &
RP=$!
m() { for i in $(seq 1 3000); do grep -q "$1" logs/$N.err 2>/dev/null && break; sleep 0.1; done; echo "$2 $(date +%H:%M:%S.%N | cut -c1-12)" >> logs/$N.marks; }
m "enc-skip: first batch" pstart
m "^prompt:" pend
wait $RP
kill $PIDS 2>/dev/null
grep -h "^prompt:\|prompt split\|enc-skip" logs/$N.err | cut -c1-120
grep -h "engram prompt\|dense prof\|xe big prof\|xe prof\|run_big" logs/$N.err | tail -12
python3 - logs/$N <<'PY'
import sys, csv, statistics as st
base = sys.argv[1]
marks = dict(l.split() for l in open(base + ".marks"))
def sec(t): h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
a, b = sec(marks["pstart"]), sec(marks["pend"])
names = ["compute%", "rd GB/s", "wr GB/s", "eu.active%", "W", "pcie.rx GB/s"]
scale = [1, 1e-6, 1e-6, 1, 1, 1e-6]
print(f"prompt window {b - a:.1f} s:")
for d in range(4):
    rows = [r for r in csv.reader(open(f"{base}.card{d}.csv")) if r and r[0][:1].isdigit() and a <= sec(r[0].strip()) <= b]
    out = []
    for i, n in enumerate(names):
        v = []
        for r in rows:
            try: v.append(float(r[i + 2]) * scale[i])
            except (ValueError, IndexError): pass
        out.append(f"{n} {st.mean(v):.1f}" if v else f"{n} -")
    print(f"  card {d} ({len(rows)} samples): " + ", ".join(out))
PY
/usr/local/sbin/larry-gpu-clock restore > /dev/null
echo done
