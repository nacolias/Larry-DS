# chunkcmp.py A B - first (layer, card) whose chunk output differs between two STRATA_XE_DUMP_CHUNK files
import sys, struct, array
def recs(path):
    d = open(path, 'rb').read(); o = 0; out = []
    while o + 12 <= len(d):
        l, c, n = struct.unpack('<iii', d[o:o+12]); o += 12
        ne = n * 5120
        out.append((l, c, n, d[o:o + 4 * ne])); o += 4 * ne
    return out
A, B = recs(sys.argv[1]), recs(sys.argv[2])
print('records', len(A), len(B))
for i, (ra, rb) in enumerate(zip(A, B)):
    if ra[:3] != rb[:3]: print('header mismatch at', i, ra[:3], rb[:3]); break
    if ra[3] != rb[3]:
        fa = array.array('f', ra[3]); fb = array.array('f', rb[3])
        nd = sum(1 for x, y in zip(fa, fb) if x != y)
        md = max(abs(x - y) for x, y in zip(fa, fb))
        rows = sorted(set(k // 5120 for k, (x, y) in enumerate(zip(fa, fb)) if x != y))
        print(f'first difference: record {i} layer {ra[0]} card {ra[1]} n {ra[2]}: {nd} values, max |diff| {md:.3e}, token rows {rows[:20]}')
        break
else: print('identical')
