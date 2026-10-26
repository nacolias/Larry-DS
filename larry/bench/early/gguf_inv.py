#!/usr/bin/env python3
"""gguf_inv.py GGUF [prefix] - tensor table of a GGUF (any type ids, incl. PTQ1_0 = 143): name, type, shape, bytes."""
import struct, sys, collections
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
# ggml block sizes (elements, bytes)
BLK = {0:(1,4),1:(1,2),8:(32,34),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),30:(1,2),42:(64,18),143:(128,28),26:(1,4),24:(1,1)}
NAMES = {0:'F32',1:'F16',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',14:'Q6_K',30:'BF16',42:'Q2_0',143:'PTQ1_0',26:'I32',24:'I8'}
rows = []
for _ in range(nt):
    name = rstr(); (nd,) = rd('I'); dims = rd('Q' * nd); (ty,) = rd('I'); (off,) = rd('Q')
    ne = 1
    for d in dims: ne *= d
    be, bb = BLK.get(ty, (1, 0))
    rows.append((name, NAMES.get(ty, str(ty)), dims, ne // be * bb))
pre = sys.argv[2] if len(sys.argv) > 2 else None
tot = collections.Counter()
for name, ty, dims, b in rows:
    if pre and not name.startswith(pre): continue
    if b < 100000 and pre: continue
    print(f"{b/1e6:9.2f} MB {ty:6s} {str(list(dims)):26s} {name}")
for name, ty, dims, b in rows:
    if 'exps' not in name and name.startswith('blk.'): tot[ty] += b
print('dense blk.* bytes by type (GB):', {k: round(v/1e9, 2) for k, v in tot.items()})
