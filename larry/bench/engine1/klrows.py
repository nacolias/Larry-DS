# klrows.py - per run: PPL over the prompt rows (target = next token), per-row KL vs a reference run, outliers
import sys
import numpy as np
V = 129280
ids = np.fromfile('p2k.i32', dtype=np.int32)
ref = np.memmap('logs/%s.logits' % sys.argv[1], dtype=np.float32, mode='r').reshape(-1, V)
def logp(x):
    x = x.astype(np.float64); x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
n = ref.shape[0]
CH = 128
lr_all = []
for i in range(0, n, CH):
    lr_all.append(logp(np.asarray(ref[i:i + CH])))
tgt = ids[1:n]
def stats(name, lr_list):
    nll = []
    for i, lp in enumerate(lr_list):
        rows = np.arange(i * CH, min(n - 1, i * CH + lp.shape[0]))
        nll.append(-lp[rows - i * CH, tgt[rows]])
    return np.concatenate(nll)
nll_ref = stats('ref', lr_all)
print('ref %-8s rows %d  PPL %.4f' % (sys.argv[1], n - 1, np.exp(nll_ref.mean())))
for name in sys.argv[2:]:
    x = np.memmap('logs/%s.logits' % name, dtype=np.float32, mode='r').reshape(-1, V)
    kl = np.empty(n)
    lx_list = []
    for i in range(0, n, CH):
        lx = logp(np.asarray(x[i:i + CH])); lr = lr_all[i // CH]
        kl[i:i + CH] = (np.exp(lr) * (lr - lx)).sum(axis=-1)
        lx_list.append(lx)
    nll = stats(name, lx_list)
    top = np.argsort(-kl)[:5]
    print('%-8s PPL %.4f (w/o row 111 %.4f vs ref %.4f)  KL mean %.5f  median %.5f  rows KL>0.1 %d  >1 %d  top %s  row111: nll %.3f (ref %.3f) argmax %d (ref %d)' % (
        name, np.exp(nll.mean()), np.exp(np.delete(nll, 111).mean()), np.exp(np.delete(nll_ref, 111).mean()), kl.mean(), np.median(kl),
        (kl > 0.1).sum(), (kl > 1).sum(), [(int(r), round(float(kl[r]), 3)) for r in top], nll[111], nll_ref[111],
        int(np.argmax(np.asarray(x[111]))), int(np.argmax(np.asarray(ref[111])))))
