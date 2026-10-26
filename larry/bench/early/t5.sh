#!/bin/bash
cd /opt/ai/dsv41
source xe-env.sh
run() { N=$1; shift; env "$@" STRATA_DENSE_REQUANT=q6_K STRATA_DEC_PROF=1 STRATA_XE_PROF=1 ./run.sh $N --ids-file p1.i32 -n 128
  echo "== $N ($*)"; grep -h "decode step\|^decode:\|DSpark: [0-9]\|pass_v split\|GGML_ASSERT\|ABORT\|rc=" logs/$N.err | tail -5; }
run r1
venv/bin/python tok.py dec - < logs/r1.out | head -c 400; echo
run rd1 STRATA_DSPARK=$SIDECAR STRATA_D3_PF=0
STRATA_DENSE_REQUANT=q6_K STRATA_DEC_PROF=1 ./run.sh r8k --ids-file p8k.i32 -n 200
echo "== r8k"; grep -h "^prompt:\|decode step\|^decode:\|rc=" logs/r8k.err | tail -4
venv/bin/python tok.py dec - < logs/r8k.out | head -c 900; echo
