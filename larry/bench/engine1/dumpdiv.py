# dumpdiv.py - per layer ||fn_a - fn_b|| / ||fn_a|| (and routed) for pairs of column dumps
import struct, sys
import numpy as np
REC = 4 + 24 + 24 + 5120 * 4 * 2
def load(p):
    b = open(p, 'rb').read(); out = {}
    for i in range(len(b) // REC):
        r = b[i * REC:(i + 1) * REC]
        out[struct.unpack_from('<i', r, 0)[0]] = (np.frombuffer(r, dtype='<f4', count=5120, offset=52).astype(np.float64),
                                                  np.frombuffer(r, dtype='<f4', count=5120, offset=52 + 20480).astype(np.float64))
    return out
names = sys.argv[1:]
D = {n: load('logs/%s.dump' % n) for n in names}
pairs = [(names[0], n) for n in names[1:]] + [(names[2], names[3])] if len(names) >= 4 else [(names[0], n) for n in names[1:]]
print('layer ' + ''.join(f'{a[-1]+"-"+b[-1]+" fn":>10}' for a, b in pairs) + '   |' + ''.join(f'{a[-1]+"-"+b[-1]+" ro":>10}' for a, b in pairs))
rel = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(x)
for l in sorted(D[names[0]]):
    print(f'{l:5d} ' + ''.join(f'{rel(D[a][l][0], D[b][l][0]):10.2e}' for a, b in pairs) + '   |' +
          ''.join(f'{rel(D[a][l][1], D[b][l][1]):10.2e}' for a, b in pairs))
