#!/bin/bash
# utilos.sh NAME CFG(old|new) IDS NTOK - one-shot run with xpu-smi on all four cards (250 ms); the prompt window
# (enc-skip / first batch .. "prompt:") and the decode window ("prompt:" .. "decode:") cut from the log's timing
N=$1; C=$2; P=$3; T=$4
cd /opt/ai/dsv41 && source xe-env.sh
OLD="STRATA_XE_PLANAR=0 STRATA_XE_SYNC=1 STRATA_SHEXP_INLINE=1 GGML_SYCL_DSV4_DEC=0 GGML_SYCL_LARRY_Q6R8=0 GGML_SYCL_LARRY_WOA=0 GGML_SYCL_LARRY_FASTMV=0 GGML_SYCL_GRAPH_CACHE=0 STRATA_XE_GRAPH=0 STRATA_ROUTER_FUSED=0 STRATA_PASS_CACHE=0 STRATA_DENSE_ASYNC=0 STRATA_XE_XDIRECT=0 STRATA_XE_CPU_ASYNC=0 GGML_SYCL_FASTMV_R8M=0 GGML_SYCL_LARRY_QREUSE=0 GGML_SYCL_LARRY_CPYBATCH=0 STRATA_ENG_ASYNC=0 GGML_SYCL_LARRY_NORM_SMALL=0 STRATA_RAW_F32=0 STRATA_SHEXP_GU=0 STRATA_HEAD_SPLIT=0 STRATA_PREREORDER=0 STRATA_XE_GU_BIG=0 STRATA_D3_PF=1 STRATA_KGATHER=0 GGML_SYCL_LARRY_F16GEMM=0 STRATA_XE_PREFETCH=0 STRATA_PIPE=0"
OLDARGS="--pf-b 1.43 --prompt-batch 8192"
if [ "$C" = old ]; then CFG="$OLD"; A="$OLDARGS"; else CFG="STRATA_D3_PF=0"; A=""; fi
F="utilization.compute,memory.read.bandwidth,memory.write.bandwidth,eu.active,power.draw,pcie.rx.throughput.kbs"
rm -f logs/$N.err logs/$N.marks
PIDS=""
for d in 0 1 2 3; do xpu-smi dump -d $d --metrics $F --loop-ms 250 > logs/$N.card$d.csv 2>&1 & PIDS="$PIDS $!"; done
env $CFG STRATA_DENSE_REQUANT=q6_K ./run.sh $N --ids-file $P -n $T $A &
RP=$!
m() { for i in $(seq 1 1200); do grep -q "$1" logs/$N.err 2>/dev/null && break; sleep 0.1; done; echo "$2 $(date +%H:%M:%S.%N | cut -c1-12)" >> logs/$N.marks; }
m "enc-skip: first batch" pstart
m "^prompt:" pend
m "^decode:" dend
wait $RP
kill $PIDS 2>/dev/null
grep -h "^prompt:\|^decode:" logs/$N.err | cut -c1-70
python3 - logs/$N <<'PY'
import sys, csv, statistics as st
base = sys.argv[1]
marks = dict(l.split() for l in open(base + ".marks"))
def sec(t): h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
win = {"prompt": (sec(marks["pstart"]), sec(marks["pend"])), "decode": (sec(marks["pend"]) + 1.0, sec(marks["dend"]))}
names = ["compute%", "rd GB/s", "wr GB/s", "eu.active%", "W", "pcie.rx GB/s"]
scale = [1, 1e-6, 1e-6, 1, 1, 1e-6]
for wn, (a, b) in win.items():
    if b - a < 1: continue
    print(f"{wn} window {b - a:.1f} s:")
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
