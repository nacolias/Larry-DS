# routercheck.py GGUF DUMP [layers] - a column dump's routing (ids / wts) against the router recomputed from its FFN
# input: sqrt(softplus(W fn)) + exp_probs_b -> top-6, weights from the unbiased scores, normalised, x 1.5.  W as the
# engine reads it (STRATA_ROUTER_F16: an F16 copy of the GGUF's F32).  Own GGUF header parse (type ids 42 / 143).
import struct, sys
import numpy as np
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(): (n,) = rd('Q'); return f.read(n).decode('utf-8', 'replace')
assert f.read(4) == b'GGUF'
ver, = rd('I'); nt, nkv = rd('QQ')
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip(t):
    if t == 8: rstr()
    elif t == 9:
        (et,) = rd('I'); (n,) = rd('Q')
        for _ in range(n): skip(et)
    else: f.read(SZ[t])
align = 32
for _ in range(nkv):
    k = rstr(); (t,) = rd('I')
    if k == 'general.alignment': (align,) = rd('I')
    else: skip(t)
info = {}
for _ in range(nt):
    name = rstr(); (nd,) = rd('I'); dims = rd('Q' * nd); (ty,) = rd('I'); (off,) = rd('Q')
    info[name] = (dims, ty, off)
base = (f.tell() + align - 1) // align * align
def f32(name):
    dims, ty, off = info[name]
    n = 1
    for d in dims: n *= d
    if ty == 0: return np.fromfile(sys.argv[1], dtype='<f4', count=n, offset=base + off)
    if ty == 1: return np.fromfile(sys.argv[1], dtype='<f2', count=n, offset=base + off).astype(np.float32)
    raise SystemExit('%s: type %d' % (name, ty))
REC = 4 + 24 + 24 + 5120 * 4 * 2
b = open(sys.argv[2], 'rb').read()
layers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
for i in range(len(b) // REC):
    r = b[i * REC:(i + 1) * REC]
    l = struct.unpack_from('<i', r, 0)[0]
    if layers and l not in layers: continue
    ids = list(struct.unpack_from('<6i', r, 4)); wts = np.array(struct.unpack_from('<6f', r, 28))
    fn = np.frombuffer(r, dtype='<f4', count=5120, offset=52).astype(np.float64)
    wname = 'blk.%d.ffn_gate_inp.weight' % l
    if wname not in info:
        print('layer %2d: no ffn_gate_inp (%s)' % (l, ', '.join(k for k in info if k.startswith('blk.%d.ffn_gate' % l))))
        continue
    W = f32(wname).astype(np.float16).astype(np.float64).reshape(-1, 5120)
    bias = f32('blk.%d.exp_probs_b.bias' % l).astype(np.float64).reshape(-1)
    sc = np.sqrt(np.logaddexp(0, W @ fn))
    s0 = sc + bias
    order = np.argsort(-s0)
    top = [int(x) for x in order[:6]]
    w = sc[top] / sc[top].sum() * 1.5
    same = sorted(top) == sorted(ids)
    wd = np.abs(np.sort(w) - np.sort(wts)).max() if same else float('nan')
    print('layer %2d: %s  6th-7th margin %.2e  max |w diff| %.1e%s' % (
        l, 'ids match' if same else 'ids DIFFER: ref %s dumped %s' % (top, ids), s0[order[5]] - s0[order[6]], wd,
        '' if same else '  (dumped ids score ranks %s)' % [int(np.where(order == e)[0][0]) for e in ids]))
