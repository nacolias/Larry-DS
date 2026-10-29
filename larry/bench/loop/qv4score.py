#!/usr/bin/env python3
"""qv4score.py RUN ... - qv4.py's dumps against the original model (qv/qv_score.py's measures): for each run
/srv/dsv41/qv4/RUN/reqNNN.tail.f32 (the rows of corpus sequence NNN after the one its request reused) and the teacher's
/srv/dsv41/qv-work/tlogits/sNN.npy, the rows both have that predict a token of the sequence (row t predicts token t+1):
KL(teacher || engine), top-1 agreement with the teacher's top token, the next token's NLL under both (-> perplexity);
per domain and overall, the runs side by side.  Run with /opt/ai/dsv41/venv/bin/python -I."""
import json
import os
import sys

import numpy as np

W = '/srv/dsv41/qv-work'
V = 129280
meta = json.load(open(os.path.join(W, 'meta.json')))
tokens = np.load(os.path.join(W, 'tokens.npy'))
N, S = tokens.shape


def logsm(x):
    m = x.max(1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(1, keepdims=True))


res = {}
for run in sys.argv[1:]:
    d = '/srv/dsv41/qv4/%s' % run
    acc = {}
    for i in range(N):
        p = os.path.join(d, 'req%03d.tail.f32' % i)
        if not os.path.exists(p):
            continue
        E = np.memmap(p, dtype=np.float32, mode='r').reshape(-1, V)
        T = np.load(os.path.join(W, 'tlogits', 's%02d.npy' % i), mmap_mode='r')
        K = E.shape[0]
        p0 = S - K                      # the dump's first row is position p0
        lo, hi = max(p0, 0), min(S - 1, len(T))   # positions with a target in the sequence and a teacher row
        dom = meta['domains'][i]
        kl_all, hits, nt, ne = [], 0, 0.0, 0.0
        for c in range(lo, hi, 256):
            r1 = min(c + 256, hi)
            lt = logsm(np.asarray(T[c:r1], np.float64))
            le = logsm(np.asarray(E[c - p0:r1 - p0], np.float64))
            kl_all.append((np.exp(lt) * (lt - le)).sum(1))
            hits += int((lt.argmax(1) == le.argmax(1)).sum())
            tgt = tokens[i, c + 1:r1 + 1]
            nt -= float(lt[np.arange(len(tgt)), tgt].sum())
            ne -= float(le[np.arange(len(tgt)), tgt].sum())
        kl = np.concatenate(kl_all)
        for key in (dom, 'ALL'):
            a = acc.setdefault(key, [[], 0, 0.0, 0.0])
            a[0].append(kl)
            a[1] += hits
            a[2] += nt
            a[3] += ne
    res[run] = {}
    for key, a in acc.items():
        kl = np.concatenate(a[0])
        n = len(kl)
        res[run][key] = (n, float(kl.mean()), float(np.median(kl)), float(np.percentile(kl, 99)), a[1] / n,
                         float(np.exp(a[2] / n)), float(np.exp(a[3] / n)))
doms = ['ALL'] + [x for x in ('en', 'zh', 'math', 'agent', 'code') if any(x in r for r in res.values())]
print('%-8s %-12s %6s %8s %7s %7s %6s %7s %7s %6s' % ('domain', 'run', 'rows', 'KL mean', 'KL med', 'KL p99', 'top1',
                                                     'ppl T', 'ppl E', 'ratio'))
for dom in doms:
    for run in sys.argv[1:]:
        if dom not in res.get(run, {}):
            continue
        n, km, kmed, kp, t1, pt, pe = res[run][dom]
        print('%-8s %-12s %6d %8.4f %7.4f %7.3f %6.3f %7.3f %7.3f %6.3f' % (dom, run, n, km, kmed, kp, t1, pt, pe, pe / pt))
