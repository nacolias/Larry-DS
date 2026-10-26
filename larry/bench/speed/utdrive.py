#!/usr/bin/env python3
"""utdrive.py NAME SIZES [--warm N] - the service's engine (larry-ds41.json's exe + args, --serve) under unitrace,
driven over its line protocol (GEN / T / DONE): a short warm-up request, one untraced fresh prompt of --warm tokens,
then a fresh prompt per SIZE (comma-separated token counts) traced alone (unitrace --resume before it, --pause after),
4 tokens out each; QUIT so the engine exits and unitrace writes logs/ut-NAME/*.json.  Fresh = its own slice of the
xolotl sources (no prefix shared with an earlier request).  Environment for the engine: UT_ENV="K=V K=V".
Engine log: logs/engine-NAME.log.  Run from abw.sh's window (the service stopped) with venv/bin/python -I."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, '/opt/ai/dsv41')
import tok  # noqa: E402

D = '/opt/ai/dsv41'
UT = '/opt/src/pti-gpu/tools/unitrace/build/unitrace'
name, sizes = sys.argv[1], [int(x) for x in sys.argv[2].split(',')]
warm = int(sys.argv[sys.argv.index('--warm') + 1]) if '--warm' in sys.argv else 16000
cfg = json.load(open(D + '/larry-ds41.json'))
out_dir = '%s/logs/ut-%s' % (D, name)
os.makedirs(out_dir, exist_ok=True)
log = '%s/logs/engine-%s.log' % (D, name)

# token slices of the xolotl sources (the same files abx.py reads), one per request, never overlapping
TK = tok.tokenizer()
need = 64 + warm + sum(sizes)
ids = []
for root, _, fs in sorted(os.walk(D + '/xolotl/src')):
    for f in sorted(fs):
        if len(ids) < need and f.endswith(('.cpp', '.hpp', '.h', '.cu')):
            try:
                ids += TK.encode(open(os.path.join(root, f), encoding='utf-8').read())
            except (UnicodeDecodeError, OSError):
                pass
print('corpus %d tokens, need %d' % (len(ids), need), flush=True)
if len(ids) < need:
    sys.exit('corpus too small')
pos = 0


def take(n):
    global pos
    s = [0] + ids[pos:pos + n - 1]   # BOS + a slice
    pos += n
    return s


env = dict(os.environ)
for kv in os.environ.get('UT_ENV', '').split():
    k, v = kv.split('=', 1)
    env[k] = v
cmd = [UT, '--session', name, '--start-paused', '--chrome-kernel-logging', '--chrome-no-thread-on-device',
       '--output-dir-path', out_dir, cfg['exe'], '--serve'] + cfg['args']
p = subprocess.Popen(cmd, cwd=cfg['cwd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, 'w'),
                     text=True, bufsize=1, env=env)
t0 = time.time()
while True:
    ln = p.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see %s)' % log)
    if ln.startswith('READY'):
        break
print('ready in %.0f s' % (time.time() - t0), flush=True)


def gen(prompt, n, tag):
    t = time.time()
    p.stdin.write('GEN %d temperature=0 ' % n + ','.join(map(str, prompt)) + '\n')
    p.stdin.flush()
    k = 0
    while True:
        ln = p.stdout.readline()
        if not ln:
            sys.exit('engine died mid-request (%s)' % tag)
        if ln.startswith('T '):
            k += 1
        elif ln.startswith('DONE') or ln.startswith('ERR'):
            break
    print('%s: prompt %d -> %d tokens in %.2f s (%s)' % (tag, len(prompt), k, time.time() - t, ln.strip()[:60]),
          flush=True)


def ut(op):
    subprocess.run([UT, '--' + op, name], check=False, stdout=subprocess.DEVNULL)


gen(take(64), 8, 'warm-short')
gen(take(warm), 4, 'warm-%d' % warm)
for n in sizes:
    time.sleep(2)
    ut('resume')
    gen(take(n), 4, 'traced-%d' % n)
    ut('pause')
p.stdin.write('QUIT\n')
p.stdin.flush()
t = time.time()
p.wait(timeout=1800)
print('engine exited rc=%d in %.0f s; trace: %s' % (p.returncode, time.time() - t, os.listdir(out_dir)), flush=True)
