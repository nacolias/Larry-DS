#!/usr/bin/env python3
"""qv4.py NAME - the quant check against the original model (qv/: DeepSeek's 1017 release teacher-forced on the CPU,
/srv/dsv41/qv-work/tlogits/sNN.npy) on the current engine: the service's engine (larry-ds41.json's exe + args, --serve;
XE_BIN / LD_LIBRARY_PATH for a staged one) reads the 18 qv/corpus sequences (2,048 tokens: en, zh, math, agent, code)
as fresh prompts over its line protocol, and dumps every row's logits (STRATA_SERVE_DUMP_TAIL=2048 ->
/srv/dsv41/qv4/NAME/reqNNN.tail.f32, the rows after the one the request reuses).  The prompt path, routing exact.  Environment
for the engine: QV_ENV="K=V K=V" (e.g. STRATA_BLOB_RQ=<file>).  bench/loop/qv4score.py scores the dumps.  Run in abw.sh's
window (the service stopped) with venv/bin/python -I."""
import json
import os
import subprocess
import sys
import time

import numpy as np

D = '/opt/ai/dsv41'
name = sys.argv[1]
cfg = json.load(open(D + '/larry-ds41.json'))
out_dir = '/srv/dsv41/qv4/%s' % name
os.makedirs(out_dir, exist_ok=True)
log = '%s/logs/engine-qv4-%s.log' % (D, name)
seqs = [np.fromfile('%s/qv/corpus/s%02d.i32' % (D, i), dtype=np.int32).tolist() for i in range(18)]
env = dict(os.environ)
for kv in os.environ.get('QV_ENV', '').split():
    k, v = kv.split('=', 1)
    env[k] = v
env.update(STRATA_SERVE_DUMP=out_dir, STRATA_SERVE_DUMP_TAIL='2048')
print('engine env:', os.environ.get('QV_ENV', '') or "(the service's)", flush=True)
p = subprocess.Popen([cfg['exe'], '--serve'] + cfg['args'], cwd=cfg['cwd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=open(log, 'w'), text=True, bufsize=1, env=env)
t0 = time.time()
while True:
    ln = p.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see %s)' % log)
    if ln.startswith('READY'):
        break
print('ready in %.0f s' % (time.time() - t0), flush=True)
meta = []
for i, ids in enumerate(seqs):
    t = time.time()
    p.stdin.write('GEN 1 temperature=0 ' + ','.join(map(str, ids)) + '\n')
    p.stdin.flush()
    while True:
        ln = p.stdout.readline()
        if not ln:
            sys.exit('engine died at sequence %d' % i)
        if ln.startswith('DONE') or ln.startswith('ERR'):
            break
    meta.append({'seq': i, 'n': len(ids), 'done': ln.strip(), 's': round(time.time() - t, 2)})
    print('s%02d: %d tokens in %.2f s (%s)' % (i, len(ids), time.time() - t, ln.strip()[:50]), flush=True)
    json.dump(meta, open(out_dir + '/meta.json', 'w'))
p.stdin.write('QUIT\n')
p.stdin.flush()
p.wait(timeout=1800)
print('engine exited rc=%d' % p.returncode, flush=True)
