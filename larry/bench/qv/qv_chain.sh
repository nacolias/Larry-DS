#!/bin/bash
# qv_chain.sh - after tbias.sh: tbias2.sh (bias 0.25, quiet box), then the shard download + CPU teacher in the
# background and the engine's logits dumps (qv_engine.sh) in the foreground.
cd /opt/ai/dsv41
until grep -q "TBIAS done" qv/tbias.log; do sleep 20; done
qv/tbias2.sh > qv/tbias2.log 2>&1
(setsid nohup qv/qv_dl.sh > qv/dl.out 2>&1 &)
(THREADS=6 setsid nohup qv/qv_teacher.sh > qv/teacher.out 2>&1 &)
qv/qv_engine.sh > qv/engine.log 2>&1
echo "CHAIN engine part done $(date +%T)" >> qv/engine.log
