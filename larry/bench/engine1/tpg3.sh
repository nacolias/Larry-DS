#!/bin/bash
# tpg3.sh - the pager off with the dense card's expert cache as large as the pager made it (STRATA_XE_DENSE_GIB
# 14 + 4.12): is a fuller card 0 alone what breaks long prompts?
cd /opt/ai/dsv41
bench/engine1/tpg.sh pg_offbig STRATA_TP_PAGE=0 STRATA_XE_DENSE_GIB=18.12
echo "TPG3 done $(date +%T)"
