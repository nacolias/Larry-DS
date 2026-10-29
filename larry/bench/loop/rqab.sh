#!/bin/bash
# rqab.sh [PFX] - optimization #4 (the coldest experts' Q3_K parts requantized to Q2_0: ds41_requant's RQ file read through
# STRATA_BLOB_RQ, patch-blobrq.py) in a window, on the staged engine (RQ_BIN, default brqbin): the quant check against
# the original model (qv4.py: the 18 corpus sequences, every row's logits) without and with the file, then the deep agent
# replay (deeprun.sh) without / with / without / with.  Scoring: qv4score.py PFX-svc PFX-base PFX-rq (after the window).  The
# service must be stopped: an abw.sh probe ("@bash bench/loop/rqab.sh").
cd /opt/ai/dsv41
P=${1:-rq}
K=${RQ_BIN:-brqbin}
F=${RQ_FILE:-/srv/dsv41/rq/rq-q20-r9600.rq}
# the service's own engine first (the staged one is a full rebuild: the two must score alike without the file)
QV_ENV="" /opt/ai/dsv41/venv/bin/python -I bench/loop/qv4.py $P-svc 2>&1 | tail -2
[ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
export XE_BIN=/opt/ai/dsv41/$K LD_LIBRARY_PATH=/opt/ai/dsv41/$K
QV_ENV="" /opt/ai/dsv41/venv/bin/python -I bench/loop/qv4.py $P-base 2>&1 | tail -3
[ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
QV_ENV="STRATA_BLOB_RQ=$F" /opt/ai/dsv41/venv/bin/python -I bench/loop/qv4.py $P-rq 2>&1 | tail -3
grep -a -h "blob requant" logs/engine-qv4-$P-rq.log
[ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
for r in off1 on1 off2 on2; do
  E=""
  [ "${r:0:2}" = on ] && E="STRATA_BLOB_RQ=$F"
  bash bench/loop/deeprun.sh $P-$r XE_BIN=$XE_BIN LD_LIBRARY_PATH=$LD_LIBRARY_PATH $E 2>&1 | grep -E "turns 1-|\] done|STILL|exited|rror|no room"
  [ -e "${AB_FLAG:-/nonexistent}" ] && exit 3
done
