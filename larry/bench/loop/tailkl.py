#!/usr/bin/env python3
"""tailkl.py REF_DIR OTHER_DIR ... - ltkl.py runs against the first: per fresh prompt, the dumped tail rows' KL(ref ||
other) (mean, p99, max), top-1 agreement, each run's perplexity and next-token top-1 over the rows whose next token is
in the prompt, rows with non-finite logits; then how far each run's greedy continuation matches the reference's.
Run with /opt/ai/dsv41/venv/bin/python -I."""
import json
import os
import sys

import numpy as np

V = 129280
CH = 128
runs = sys.argv[1:]
metas = [json.load(open(os.path.join(r, 'meta.json'))) for r in runs]


def logp(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def load(r, k):
    a = np.memmap(os.path.join(r, 'req%03d.tail.f32' % k), dtype=np.float32, mode='r')
    return a.reshape(-1, V)


def own(a, tgt, n):   # perplexity and next-token top-1 over rows 0 .. n-2 (row i predicts tgt[i]), non-finite rows
    s, hit, bad = 0.0, 0, 0
    for i in range(0, n - 1, CH):
        x = np.asarray(a[i:min(n - 1, i + CH)])
        bad += int((~np.isfinite(x)).any(axis=-1).sum())
        lp = logp(x)
        t = tgt[i:i + lp.shape[0]]
        s -= lp[np.arange(lp.shape[0]), t].sum()
        hit += int((lp.argmax(-1) == t).sum())
    return float(np.exp(s / (n - 1))), 100.0 * hit / (n - 1), bad


for k, m in enumerate(metas[0]):
    if not m['tag'].startswith('fresh'):
        continue
    ids = np.array(m['tail_ids'], dtype=np.int64)
    ref = load(runs[0], k)
    n = ref.shape[0]
    if n != len(ids):
        sys.exit('%s: %d rows dumped, %d ids' % (m['tag'], n, len(ids)))
    tgt = ids[1:]
    print('%s (%d tokens): the last %d rows' % (m['tag'], m['size'], n))
    print('  %-24s PPL %.4f  next-token top-1 %.2f %%  non-finite rows %d' % ((os.path.basename(runs[0]),) + own(ref, tgt, n)))
    for r, mm in zip(runs[1:], metas[1:]):
        if mm[k]['tail_ids'] != m['tail_ids']:
            sys.exit('%s: %s read other ids' % (m['tag'], r))
        b = load(r, k)
        kl, top = [], 0
        for i in range(0, n, CH):
            la = logp(np.asarray(ref[i:min(n, i + CH)]))
            lb = logp(np.asarray(b[i:min(n, i + CH)]))
            kl.extend((np.exp(la) * (la - lb)).sum(axis=-1).tolist())
            top += int((la.argmax(-1) == lb.argmax(-1)).sum())
        kl = np.array(kl)
        print('  %-24s PPL %.4f  next-token top-1 %.2f %%  non-finite rows %d' % ((os.path.basename(r),) + own(b, tgt, n)))
        print('  %-24s KL mean %.2e  p99 %.2e  max %.2e  top-1 agreement %.2f %%'
              % ('  vs ' + os.path.basename(runs[0]), kl.mean(), np.percentile(kl, 99), kl.max(), 100.0 * top / n))
    g0 = m['gen']
    for r, mm in zip(runs[1:], metas[1:]):
        g = mm[k]['gen']
        d = next((i for i, (x, y) in enumerate(zip(g0, g)) if x != y), min(len(g0), len(g)))
        print('  greedy continuation of %s: the same as %s\'s for %d of %d tokens'
              % (os.path.basename(r), os.path.basename(runs[0]), d, len(g0)))
