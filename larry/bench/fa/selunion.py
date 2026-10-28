#!/usr/bin/env python3
"""selunion.py logs/dump_sel.PID.bin [MAXREC] - how much consecutive queries' compressed-row selections overlap
(STRATA_DUMP_SEL, patch-dumpsel.py): per record (one index source layer of one prompt batch) and block size Bq, the
mean / p90 / max size of the union of the valid selected rows of Bq consecutive columns, and the key count a
block-union attention would run per query ((union + Bq + 127 window rows) per block of Bq queries, against 640 a query).
"""
import struct
import sys

import numpy as np

path = sys.argv[1]
maxrec = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 30
BQS = (1, 2, 4, 8, 16, 32, 64, 128)
f = open(path, 'rb')
nrec = 0
agg = {}
while nrec < maxrec:
    hd = f.read(12)
    if len(hd) < 12:
        break
    layer, B, itop = struct.unpack('<iii', hd)
    p0, = struct.unpack('<q', f.read(8))
    sel = np.frombuffer(f.read(4 * B * itop), dtype=np.int32).reshape(B, itop)
    msk = np.frombuffer(f.read(4 * B * itop), dtype=np.float32).reshape(B, itop)
    nrec += 1
    valid = msk == 0
    nv = valid.sum(1)
    line = ['layer %2d pos %7d..%7d: valid a query %.0f (min %d)' % (layer, p0, p0 + B - 1, nv.mean(), nv.min())]
    for bq in BQS:
        us = []
        for b0 in range(0, B - bq + 1, bq):
            ids = sel[b0:b0 + bq][valid[b0:b0 + bq]]
            us.append(len(np.unique(ids)))
        us = np.array(us, dtype=np.float64)
        keys_q = (us.mean() + bq + 127 + 1) / 1.0   # a block's keys (each of its bq queries runs over all of them)
        line.append('  Bq %3d: union mean %6.0f p90 %6.0f max %6d  -> %5.0f keys a query-block (x%.2f of 641)'
                    % (bq, us.mean(), np.percentile(us, 90), us.max(), keys_q, keys_q / 641.0))
        a = agg.setdefault((layer, bq), [])
        a.append(us.mean())
    print('\n'.join(line))
print('\nmean over records (key count a block runs, x of 641):')
for bq in BQS:
    row = []
    for layer in sorted(set(k[0] for k in agg)):
        u = np.mean(agg[(layer, bq)])
        row.append('L%d %6.0f (x%.2f)' % (layer, u, (u + bq + 128) / 641.0))
    print('  Bq %3d: %s' % (bq, '   '.join(row)))
