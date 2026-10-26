#!/usr/bin/env python3
# dumpcol.py A B - engine1 debug: two STRATA_DUMP_COL dumps (per layer: int32 layer, int32 ids[K], f32 wts[K],
# f32 fn[D], f32 routed[D]) compared layer by layer - where one run's column first leaves the other's
import sys, struct
import numpy as np
K, D = 6, 5120
REC = 4 + 4 * K + 4 * K + 4 * D + 4 * D
def load(p):
    b = open(p, 'rb').read()
    out = {}
    for o in range(0, len(b) - REC + 1, REC):
        l = struct.unpack_from('<i', b, o)[0]
        ids = np.frombuffer(b, np.int32, K, o + 4)
        wts = np.frombuffer(b, np.float32, K, o + 4 + 4 * K)
        fn = np.frombuffer(b, np.float32, D, o + 4 + 8 * K)
        rt = np.frombuffer(b, np.float32, D, o + 4 + 8 * K + 4 * D)
        out[l] = (ids, wts, fn, rt)
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
for l in sorted(set(a) & set(b)):
    ia, wa, fa, ra = a[l]
    ib, wb, fb, rb = b[l]
    rel = lambda x, y: float(np.abs(x - y).max() / (np.abs(x).max() + 1e-30))
    same_ids = sorted(ia.tolist()) == sorted(ib.tolist())
    print("layer %2d  ids %s  wts %.2e  fn %.2e  routed %.2e%s" % (l, "same" if same_ids else "DIFF %s vs %s" % (ia.tolist(), ib.tolist()),
          float(np.abs(np.sort(wa) - np.sort(wb)).max()), rel(fa, fb), rel(ra, rb), "   <==" if (not same_ids or rel(fa, fb) > 1e-2 or rel(ra, rb) > 1e-2) else ""))
