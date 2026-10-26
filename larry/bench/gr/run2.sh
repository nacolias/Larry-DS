#!/bin/bash
# run2.sh - GROUTE: tgr.sh (check + A/B + timelines), then tgr2.sh (the splits on top), JOBNT=1
cd /opt/ai/dsv41/bench/gr
echo "== tgr $(date +%T)"; JOBNT=1 ./tgr.sh > tgr.log 2>&1; echo "tgr rc=$?"
echo "== tgr2 $(date +%T)"; JOBNT=1 ./tgr2.sh > tgr2.log 2>&1; echo "tgr2 rc=$?"
echo "ALL2 done $(date +%T)"
