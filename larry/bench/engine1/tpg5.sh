#!/bin/bash
# tpg5.sh - the pager with the fix (the fills' card-to-card copies pulled by card 0 on its own copy queue), the
# service sequence twice in one server
cd /opt/ai/dsv41
TPG_ROUNDS=2 bench/engine1/tpg.sh pg_fix STRATA_TP_PAGE=1
echo "TPG5 done $(date +%T)"
