#!/bin/bash
# r1m-s2.sh - the lazy card-0 KV map's garbled 1M decode (r1m.sh variant A: a 131K needle into conversation slot 1, then a
# 984K needle into slot 0) on the staged engine (s2bin) with patch-tpsynclog.py's sync log + replica check and the TP
# core's per-pass profile.  An abw.sh "@" step.
cd /opt/ai/dsv41
R1M_VARIANTS=A R1M_TOKENS=131072,984000 \
  R1M_ENV="XE_BIN=/opt/ai/dsv41/s2bin LD_LIBRARY_PATH=/opt/ai/dsv41/s2bin STRATA_TP_SYNC_LOG=1 STRATA_TP_SYNC_CHECK=1 STRATA_TP_PROF=1" \
  bash bench/conc/r1m.sh
