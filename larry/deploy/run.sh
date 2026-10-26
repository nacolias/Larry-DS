#!/bin/bash
# run.sh NAME [engine args...] - one-shot ds41_generate_xe run; logs/NAME.{out,err}.  Env passes through.
# Runs are serialised on /run/lock/ds41-oneshot.lock (one engine per box: VRAM, pinned memory).
#   DS41_UNIT (ds41; larry: larry-ds41): the service that must be stopped first   RUN_MEM_MAX (112G: larry has 121 GiB)
N=$1; shift
ARGS=("$@")
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$D" && source xe-env.sh
# one engine per box: the service holds every card (and ~40 GiB of driver memory) - its watchdog killed it when a
# one-shot run started beside it (2026-10-20 20:40)
if systemctl is-active --quiet "${DS41_UNIT:-ds41}"; then echo "run.sh: ${DS41_UNIT:-ds41} is running - stop it first" >&2; exit 3; fi
flock -w 7200 /run/lock/ds41-oneshot.lock timeout ${RUN_TIMEOUT:-1500} systemd-run --scope -q -p MemoryMax=${RUN_MEM_MAX:-112G} -p MemorySwapMax=0 \
  $XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null
echo "rc=$?" >> logs/$N.err
