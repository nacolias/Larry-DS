#!/bin/bash
# r1m-stage.sh - patch-tpstage.py on the staged engine: the short lazy-KV sequence with the replica check (lazyab.sh
# base: a 131K needle, its turn), then the 984K reproduction (r1m-s2.sh: a 131K needle into conversation slot 1, a 984K
# needle into slot 0, with the sync log + check and the per-pass profile).  An abw.sh "@" step.
cd /opt/ai/dsv41
LAZYAB_CONFIGS=base bash bench/conc/lazyab.sh
bash bench/conc/r1m-s2.sh
