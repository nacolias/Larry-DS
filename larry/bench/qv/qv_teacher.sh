#!/bin/bash
# qv_teacher.sh - the ORIGINAL DeepSeek-V4.1-Flash-1017 (FP8 dense, FP4 experts, as DeepSeek ships it) teacher-forced
# over qv/corpus on larry's CPU: DeepSeek's own inference/model.py on Xolotl's torch stand-in kernels, one layer at a
# time straight from the shards (Xolotl's v41_calib.py), then the head -> full logits per sequence.
#   needs: qv_dl.sh downloading into /srv/dsv41/orig, qv_engram_small.py's two files there, the engine's extracted
#   Engram tables linked as engram-L1 / engram-L14 in /opt/ai/dsv41/qv/engram.  Each layer's shard is deleted after
#   the layer is done (qv_dl.sh keeps <= 10 on disk).  Resumable: a layer is done once stats/LNN.json exists.
set -u
W=/srv/dsv41/qv-work
export V41_WORK=$W V41_SRC="/opt/ai/dsv41/qv/ds1017:/srv/dsv41/orig:/opt/ai/dsv41/qv/engram"
export OMP_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8}
PY=/opt/ai/dsv41/tvenv/bin/python
CAL=/opt/ai/dsv41/qv/qv_calib.py   # v41_calib.py main() with its missing ABL import stubbed
Q=/opt/ai/dsv41/qv
mkdir -p $W/logs
log() { echo "$(date '+%F %T') $*" | tee -a $W/teacher.log; }
ok() { for d in /srv/dsv41/orig /opt/ai/dsv41/qv/engram; do [ -f "$d/$1.ok" ] && return 0; done; return 1; }
waitf() { until ok "$1"; do sleep 20; done; }
cd $Q
[ -f $W/hashes.npy ] || { log "prep"; nice -n 5 $PY $Q/qv_prep.py $Q/corpus > $W/logs/prep.log 2>&1 || { log "prep failed"; exit 1; }; }
if [ ! -f $W/h/L00.bin ] && [ ! -f $W/stats/L00.json ]; then
  waitf model-00002-of-00048.safetensors
  log "init"
  nice -n 5 $PY $CAL init > $W/logs/init.log 2>&1 || { log "init failed"; exit 1; }
  rm -f /srv/dsv41/orig/model-00002-of-00048.safetensors
fi
for L in $(seq 0 39); do
  N=$(printf %02d $L); f=model-$(printf %05d $((L + 3)))-of-00048.safetensors
  [ -f $W/stats/L$N.json ] && { rm -f /srv/dsv41/orig/$f; continue; }
  waitf $f
  log "layer $L start"
  nice -n 5 $PY $CAL layer $L --device cpu --no-gptq --gc > $W/logs/L$N.log 2>&1 || { log "layer $L FAILED (logs/L$N.log)"; exit 1; }
  log "layer $L done: $(grep -o "done in [0-9]* s.*" $W/logs/L$N.log | tail -1)"
  rm -f /srv/dsv41/orig/$f
done
waitf model-00043-of-00048.safetensors
log "head"
nice -n 5 $PY $Q/qv_head.py $W/tlogits > $W/logs/head.log 2>&1 || { log "head failed"; exit 1; }
grep "head: [A-Za-z]" $W/logs/head.log | tee -a $W/teacher.log
log "TEACHER DONE"
