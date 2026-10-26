# dumpnorm.py - per layer |fn|, |routed| of each dump (column dumps from STRATA_DUMP_COL), plus ids / wts of layers asked for
import struct, sys, math
import numpy as np
REC = 4 + 24 + 24 + 5120 * 4 * 2
def load(p):
    b = open(p, 'rb').read()
    out = {}
    for i in range(len(b) // REC):
        r = b[i * REC:(i + 1) * REC]
        l = struct.unpack_from('<i', r, 0)[0]
        ids = struct.unpack_from('<6i', r, 4)
        wts = struct.unpack_from('<6f', r, 28)
        fn = np.frombuffer(r, dtype='<f4', count=5120, offset=52)
        ro = np.frombuffer(r, dtype='<f4', count=5120, offset=52 + 20480)
        out[l] = (ids, wts, fn, ro)
    return out
paths = sys.argv[1].split(',')
show = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else []
D = [load(p) for p in paths]
print('layer ' + ''.join(f'{"run"+str(i+1)+" |fn| |ro| max|ro|":>34}' for i in range(len(D))))
for l in sorted(D[0]):
    s = f'{l:5d} '
    for d in D:
        ids, wts, fn, ro = d[l]
        s += f'{np.linalg.norm(fn):10.3f} {np.linalg.norm(ro):10.3f} {np.abs(ro).max():10.3f}  '
    print(s)
for l in show:
    print(f'-- layer {l}')
    for i, d in enumerate(D):
        ids, wts, fn, ro = d[l]
        am = int(np.abs(ro).argmax())
        print(f'  run{i+1} ids {list(ids)} wts {[round(w,4) for w in wts]} sum {sum(wts):.4f} argmax|ro| {am} ({ro[am]:.3f}) nan {int(np.isnan(ro).sum())}')
