# dumphyp.py - the bad run's layer-L routed output minus the good runs': what does the difference look like?
import struct, sys
import numpy as np
REC = 4 + 24 + 24 + 5120 * 4 * 2
def load(p):
    b = open(p, 'rb').read(); out = {}
    for i in range(len(b) // REC):
        r = b[i * REC:(i + 1) * REC]
        l = struct.unpack_from('<i', r, 0)[0]
        out[l] = (struct.unpack_from('<6i', r, 4), struct.unpack_from('<6f', r, 28),
                  np.frombuffer(r, dtype='<f4', count=5120, offset=52).astype(np.float64),
                  np.frombuffer(r, dtype='<f4', count=5120, offset=52 + 20480).astype(np.float64))
    return out
bad = load(sys.argv[1]); goods = [load(p) for p in sys.argv[2].split(',')]; L = int(sys.argv[3])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
ref = np.mean([g[L][3] for g in goods], axis=0)
d = bad[L][3] - ref
print(f'layer {L}: |bad| {np.linalg.norm(bad[L][3]):.3f} |good mean| {np.linalg.norm(ref):.3f} |d| {np.linalg.norm(d):.3f}')
print('  good-run spread at L:', [round(float(np.linalg.norm(g[L][3] - ref)), 3) for g in goods])
top = np.argsort(-np.abs(d))[:12]
print('  top |d| channels:', [(int(c), round(float(d[c]), 3)) for c in top])
print('  share of |d|^2 in top 1/4/16 channels:', [round(float(np.sort(d**2)[::-1][:k].sum() / (d**2).sum()), 3) for k in (1, 4, 16)])
for name, v in [('bad ro L-1', bad[L-1][3]), ('good ro L-1', goods[0][L-1][3]), ('bad ro L-2', bad[L-2][3]),
                ('bad fn L', bad[L][2]), ('bad fn L-1', bad[L-1][2]), ('good ro L', ref), ('bad ro L+1', bad[L+1][3])]:
    print(f'  cos(d, {name:12s}) {cos(d, v):+.3f}   lstsq scale {float(d @ v / (v @ v)):+.3f}')
# d without channel 2455 / the top channel
m = np.ones(5120, bool); m[top[0]] = False
print(f'  |d| without ch {top[0]}: {np.linalg.norm(d[m]):.3f}; good spread without it:',
      [round(float(np.linalg.norm((g[L][3] - ref)[m])), 3) for g in goods])
