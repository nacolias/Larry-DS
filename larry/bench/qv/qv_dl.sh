#!/bin/bash
# qv_dl.sh - DeepSeek-V4.1-Flash-1017's shards 2..43 (embedding, the 40 layers, final norm + head; 298 GB) into
# /srv/dsv41/orig, three at a time, each checked against the Hub's LFS sha256 before its .ok marker is written.
# At most MAXON shards (default 10, ~74 GB) sit on disk unprocessed: qv_teacher.sh deletes a layer's shard once the
# layer is done, and this waits.  Resumable (curl -C -; a shard with its .ok is skipped).
set -u
D=/srv/dsv41/orig; MAXON=${MAXON:-10}
mkdir -p $D
cd $D
URL=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main
[ -s lfs.json ] || curl -s "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main" > lfs.json
sha_of() { /opt/ai/dsv41/venv/bin/python -c "import json,sys; print(next(f['lfs']['oid'] for f in json.load(open('lfs.json')) if f['path']==sys.argv[1]))" "$1"; }
log() { echo "$(date '+%F %T') $*" >> $D/dl.log; }
one() {
  f=$1; want=$(sha_of $f)
  for try in 1 2 3 4; do
    curl -sfL --retry 5 --retry-delay 10 -C - -o $f.part "$URL/$f"
    got=$(sha256sum $f.part | cut -d" " -f1)
    if [ "$got" = "$want" ]; then mv $f.part $f; touch $f.ok; log "ok $f ($(stat -c %s $f) bytes)"; return 0; fi
    log "sha mismatch $f (try $try): $got vs $want"; rm -f $f.part
  done
  log "FAILED $f"; return 1
}
for i in $(seq 2 43); do
  f=model-$(printf %05d $i)-of-00048.safetensors
  [ -f $f.ok ] && continue
  # throttle: shards on disk (downloaded or downloading) that the teacher has not consumed yet
  while [ "$(ls $D | grep -c '\.safetensors\(\.part\)\?$')" -ge "$MAXON" ]; do sleep 20; done
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
  one $f &
  sleep 2
done
wait
log "ALL DONE"
