#!/bin/bash
# rungdb.sh NAME [engine args...] - like run.sh, under gdb (backtrace on a crash)
N=$1; shift
ARGS=("$@")
cd /opt/ai/dsv41 && source xe-env.sh
timeout ${RUN_TIMEOUT:-1500} systemd-run --scope -q -p MemoryMax=112G -p MemorySwapMax=0 \
  gdb -q -batch -ex "handle SIGUSR1 SIGUSR2 SIGPIPE nostop noprint" -ex run -ex "thread apply all bt 12" --args \
  $XE_BIN/ds41_generate_xe -m $GGUF --backend gpu --experts gpu --engram $ENGRAM --rank $RANK "${ARGS[@]}" > logs/$N.out 2> logs/$N.err < /dev/null
echo "rc=$?" >> logs/$N.err
