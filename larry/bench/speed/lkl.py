#!/usr/bin/env python3
"""lkl.py IDS.i32 REF.logits OTHER.logits ... - prompt-path logits dumps (--dump-logits: f32 [rows, V]) against the
first: each run's perplexity over the prompt (row i predicts token i + 1), and per row KL(ref || other), top-1
agreement.  Run with /opt/ai/dsv41/venv/bin/python -I."""
import sys
import numpy as np
V = 129280
ids = np.fromfile(sys.argv[1], dtype=np.int32)


def load(p):
    a = np.memmap(p, dtype=np.float32, mode='r')
    return a.reshape(-1, V)


def logp(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


ref = load(sys.argv[2])
n = min(ref.shape[0], len(ids)) - 1
tgt = ids[1:n + 1]
CH = 128


def ppl(m):
    s = 0.0
    for i in range(0, n, CH):
        lp = logp(np.asarray(m[i:min(n, i + CH)]))
        s -= lp[np.arange(lp.shape[0]), tgt[i:i + lp.shape[0]]].sum()
    return float(np.exp(s / n))


print('%-40s rows %d  PPL %.4f' % (sys.argv[2], n, ppl(ref)))
for p in sys.argv[3:]:
    b = load(p)
    kl, top = [], 0
    for i in range(0, n, CH):
        la = logp(np.asarray(ref[i:min(n, i + CH)]))
        lb = logp(np.asarray(b[i:min(n, i + CH)]))
        kl.extend((np.exp(la) * (la - lb)).sum(axis=-1).tolist())
        top += int((la.argmax(-1) == lb.argmax(-1)).sum())
    kl = np.array(kl)
    print('%-40s PPL %.4f  KL mean %.2e  p99 %.2e  max %.2e  top-1 %.2f %%' % (
        p, ppl(b), kl.mean(), np.percentile(kl, 99), kl.max(), 100.0 * top / n))
