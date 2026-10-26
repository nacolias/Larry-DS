#!/bin/bash
# tr12.sh - engine1 replicas, step 5: ggml-sycl's reclaim hook (a full card asks the engine for VRAM back before an
# allocation fails: the tier drops the dense card's replicas).  Rebuild the engine (ggml-sycl's common.cpp /
# ggml-sycl.cpp / ggml-sycl.h changed) and the libs; then the service's test sequence on a private port with the
# replicas on: twice at the default 1.0 GiB margin (the 21:41 crash: an Engram head split's pool growth in the 4th
# svcbench request), once at 0.1 GiB (every growth reclaims).  Service held.
cd /opt/ai/dsv41/bench/engine1
t0=$(date +%s)
TARGETS="ds41_generate_xe ds41_xe ds41_tp tp_bench" ./build-tier.sh || { echo "BUILD FAILED"; exit 1; }
echo "build $(( $(date +%s) - t0 )) s"
cd /opt/ai/dsv41
REP="STRATA_XE_REPLICA_GIB=2.5 STRATA_XE_REPLICA_USAGE=/opt/ai/dsv41/replica-usage.txt"
TPG_ROUNDS=2 bench/engine1/tpg.sh rep2 $REP 2>&1
tr "\r" "\n" < logs/engine-rep2.log | grep -a "replicas: [0-9]* on\|replicas given back\|Failed to allocate\|SYCL error" | sort | uniq -c | sort -rn | head -8
TPG_ROUNDS=1 bench/engine1/tpg.sh rep01 $REP STRATA_XE_REPLICA_MARGIN_GIB=0.1 2>&1
tr "\r" "\n" < logs/engine-rep01.log | grep -a "replicas: [0-9]* on\|replicas given back\|Failed to allocate\|SYCL error" | sort | uniq -c | sort -rn | head -8
dmesg -T | grep -c "CAT error"
echo "TR12 done $(date +%T)"
