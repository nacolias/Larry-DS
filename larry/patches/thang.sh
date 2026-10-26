#!/bin/bash
# thang.sh - one 8.5K pipelined prompt; if it stalls 90 s after the batch starts: dump every thread, kill, GT-reset
# all four cards (a kill with copies in flight faults the copy engines) and re-probe them
cd /opt/ai/dsv41
rm -f logs/ph.err logs/ph.gdb
env "$@" STRATA_DENSE_REQUANT=q6_K RUN_TIMEOUT=600 ./run.sh ph --ids-file ${PF:-p8k.i32} -n 4 $EARGS &
RP=$!
for i in $(seq 1 300); do grep -q "xe prefetch: layers\|enc-skip: first batch" logs/ph.err 2>/dev/null && break; sleep 1; done
for i in $(seq 1 ${HANG_S:-90}); do grep -q "^prompt:" logs/ph.err 2>/dev/null && break; sleep 1; done
P=$(pgrep -x ds41_generate_x | head -1)
if [ -n "$P" ] && ! grep -q "^prompt:" logs/ph.err; then
  echo "hung: dumping $P"
  timeout 120 gdb -p $P -batch -ex "thread apply all bt 16" > logs/ph.gdb 2>&1
  kill -9 $P; sleep 8
  for p in 0000:15:00.0 0000:1b:00.0 0000:22:00.0 0000:26:00.0; do cat /sys/kernel/debug/dri/$p/gt0/force_reset_sync > /dev/null 2>&1; done
  sleep 3
  (cd bench && source /opt/intel/oneapi/2026.1/oneapi-vars.sh >/dev/null 2>&1; for d in 0 1 2 3; do echo "probe $d: $(ONEAPI_DEVICE_SELECTOR=level_zero:$d timeout 60 ./copy_probe 2>&1 | tail -1)"; done)
else
  echo "no hang"
fi
wait $RP
grep -h "^prompt:\|rc=" logs/ph.err | tail -2
