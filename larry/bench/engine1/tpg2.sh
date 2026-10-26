#!/bin/bash
# tpg2.sh - which part breaks long prompts in the service with the pager on: (1) the pager as deployed (does the
# private-port harness reproduce it?), (2) the pager off with the dense card's expert cache as large as the pager made
# it (STRATA_XE_DENSE_GIB 14 + 4.12)
cd /opt/ai/dsv41
bench/engine1/tpg.sh pg_on STRATA_TP_PAGE=1
bench/engine1/tpg.sh pg_offbig STRATA_TP_PAGE=0 STRATA_XE_DENSE_GIB=18.12
echo "TPG2 done $(date +%T)"
