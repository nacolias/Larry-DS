import sys, numpy as np
# lcmp.py A.f32 B.f32 ... - compare last-row logits dumps against the first: max |d|, KL(A||X), top-1, top-20 overlap
def load(p): return np.fromfile(p, dtype=np.float32).astype(np.float64)
def sm(x): x = x - x.max(); e = np.exp(x); return e / e.sum()
a = load(sys.argv[1]); pa = sm(a); ta = np.argsort(-a)[:20]
for p in sys.argv[2:]:
    b = load(p); pb = sm(b); tb = np.argsort(-b)[:20]
    kl = float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))
    print("%-28s max|d| %.4f  KL %.2e  top1 %s  top20 overlap %d  p(top1) %.4f vs %.4f" % (p, np.abs(a - b).max(), kl, ta[0] == tb[0], len(set(ta) & set(tb)), pa[ta[0]], pb[ta[0]]))
