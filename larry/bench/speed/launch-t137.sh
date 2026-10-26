#!/bin/bash
# launch-t137.sh - abwT137: STRATA_DSPARK_CUM=0.5 (patch-dsparkcum.py: verify a draft only while the product of the
# drafts' confidences so far stays >= 0.5, besides STRATA_DSPARK_CONF's 0.3 floor on each; Project Maya's
# MAYA_DS4_DSPARK_CUM) against off, three fresh servers each, interleaved; the decode tasks (essay, code, math; 400 out).
cd /opt/ai/dsv41
AB_IDLE_MIN=0 setsid nohup bash bench/speed/abw.sh c0:dec c5:dec:STRATA_DSPARK_CUM=0.5 c0b:dec c5b:dec:STRATA_DSPARK_CUM=0.5 \
  c0c:dec c5c:dec:STRATA_DSPARK_CUM=0.5 > logs/abwT137.log 2>&1 < /dev/null &
echo "abwT137 launched: logs/abwT137.log"
