#!/bin/bash
# dsel.sh - an abw.sh probe (#3): the staged dselbin engine (patch-dumpsel.py) with STRATA_DUMP_SEL=1 through p30k
# (code, 31.6K tokens) and pread (the engineering log as prose, 94K): every batch's index-source selections (layers 2 /
# 8 / 14) -> logs/dump_sel_<IDS>.bin, summarized by bench/fa/selunion.py - how many distinct compressed rows blocks of
# Bq consecutive queries select (a block-union attention's key count).
cd /opt/ai/dsv41
eval "$(grep '^export ' ds41-xe-engine.sh)"
unset STRATA_SERVE_DSPARK
for IDS in p30k pread; do
  rm -f logs/dump_sel.*.bin
  STRATA_DUMP_SEL=1 XE_BIN=/opt/ai/dsv41/dselbin LD_LIBRARY_PATH=/opt/ai/dsv41/dselbin RUN_TIMEOUT=1500 ./run.sh dsel_$IDS --ids-file $IDS.i32 -n 1 --prompt-batch 16384
  grep -a -h "^prompt\|STRATA_DUMP_SEL\|^rc=" logs/dsel_$IDS.err | tail -10
  F=$(ls -t logs/dump_sel.*.bin 2>/dev/null | head -1)
  [ -n "$F" ] && mv $F logs/dump_sel_$IDS.bin
done
