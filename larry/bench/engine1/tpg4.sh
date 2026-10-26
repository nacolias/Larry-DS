#!/bin/bash
# tpg4.sh - the pager on, two ways around the remapping: STRATA_TP_PAGE_ON0=1 (the copies on card 0's queue: peer
# READS into card 0's own mapping instead of the other cards' peer WRITES into it) and STRATA_TP_PAGE_KEEP=1 (no
# remapping at all: the load pages kept, every unit refilled in place each pass)
cd /opt/ai/dsv41
bench/engine1/tpg.sh pg_on0 STRATA_TP_PAGE=1 STRATA_TP_PAGE_ON0=1
bench/engine1/tpg.sh pg_keep STRATA_TP_PAGE=1 STRATA_TP_PAGE_KEEP=1
echo "TPG4 done $(date +%T)"
