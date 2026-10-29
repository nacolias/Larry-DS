#!/usr/bin/env python3
"""ltkl.py NAME [--sizes 96000,150000] [--tail 2048] [--gen 192] - long-context numerics of the service's engine
(larry-ds41.json's exe + args, --serve) driven over its line protocol (GEN / T / DONE): a short warm-up request, then a
fresh prompt per SIZE (its own slice of the xolotl sources: no prefix shared with an earlier request), --gen tokens
greedy after each.  The engine dumps every request's last --tail rows of prompt logits (STRATA_SERVE_DUMP_TAIL into
logs/lt-NAME/reqNNN.tail.f32, [tail x vocab]) and compares the TP replicas with card 0 after each sync
(STRATA_TP_SYNC_CHECK=1: "tp sync check" lines in the engine log).  logs/lt-NAME/meta.json: per request its size, the
ids of the dumped rows (the prompt's last `tail`), the generated ids and the timing (bench/loop/tailkl.py compares
runs; requests are reqNNN in this order: warm-short, then the sizes).  Environment for the
engine: LT_ENV="K=V K=V".  Engine log: logs/engine-lt-NAME.log.  Run in abw.sh's window (the service stopped) with
venv/bin/python -I."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, '/opt/ai/dsv41')
import tok  # noqa: E402

D = '/opt/ai/dsv41'
name = sys.argv[1]


def opt(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


sizes = [int(x) for x in opt('--sizes', '96000,150000').split(',')]
tail, ngen = int(opt('--tail', '2048')), int(opt('--gen', '192'))
cfg = json.load(open(D + '/larry-ds41.json'))
out_dir = '%s/logs/lt-%s' % (D, name)
os.makedirs(out_dir, exist_ok=True)
log = '%s/logs/engine-lt-%s.log' % (D, name)

# token slices of the xolotl sources (as utdrive.py), one per request, never overlapping
TK = tok.tokenizer()
need = 64 + sum(sizes)
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
for kv in os.environ.get('LT_ENV', '').split():
    k, v = kv.split('=', 1)
    env[k] = v
env.update(STRATA_SERVE_DUMP=out_dir, STRATA_SERVE_DUMP_TAIL=str(tail), STRATA_TP_SYNC_CHECK='1')
print('engine env:', os.environ.get('LT_ENV', '') or '(the service\'s)', flush=True)
p = subprocess.Popen([cfg['exe'], '--serve'] + cfg['args'], cwd=cfg['cwd'], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open(log, 'w'), text=True, bufsize=1, env=env)
t0 = time.time()
while True:
    ln = p.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see %s)' % log)
    if ln.startswith('READY'):
        break
print('ready in %.0f s' % (time.time() - t0), flush=True)
meta = []


def gen(prompt, n, tag):
    t = time.time()
    p.stdin.write('GEN %d temperature=0 ' % n + ','.join(map(str, prompt)) + '\n')
    p.stdin.flush()
    out, first = [], None
    while True:
        ln = p.stdout.readline()
        if not ln:
            sys.exit('engine died mid-request (%s)' % tag)
        if ln.startswith('T '):
            out.append(int(ln.split()[1]))
            if first is None:
                first = time.time() - t
        elif ln.startswith('DONE') or ln.startswith('ERR'):
            break
    print('%s: prompt %d -> %d tokens, first after %.2f s, all in %.2f s (%s)'
          % (tag, len(prompt), len(out), first or 0, time.time() - t, ln.strip()[:60]), flush=True)
    meta.append({'tag': tag, 'size': len(prompt), 'tail_ids': prompt[-tail:], 'gen': out,
                 'first_s': round(first or 0, 3), 'done': ln.strip()})
    json.dump(meta, open(out_dir + '/meta.json', 'w'))   # after every request: an engine that hangs at QUIT keeps it


gen(take(64), 8, 'warm-short')
for n in sizes:
    gen(take(n), ngen, 'fresh-%d' % n)
p.stdin.write('QUIT\n')
p.stdin.flush()
t = time.time()
p.wait(timeout=1800)
print('engine exited rc=%d in %.0f s; dumps: %s' % (p.returncode, time.time() - t, sorted(os.listdir(out_dir))),
      flush=True)
