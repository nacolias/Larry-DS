# dumpcmp.py A B n_embd - first decode step where two STRATA_DUMP_MAIN files differ, and by how much
import sys, struct, array
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read(); E = int(sys.argv[3])
rec = 8 + 3 * E * 4
n = min(len(a), len(b)) // rec
print('steps', len(a) // rec, len(b) // rec)
for i in range(n):
    ra, rb = a[i*rec:(i+1)*rec], b[i*rec:(i+1)*rec]
    if ra != rb:
        ta, pa = struct.unpack('<ii', ra[:8]); tb, pb = struct.unpack('<ii', rb[:8])
        fa = array.array('f', ra[8:]); fb = array.array('f', rb[8:])
        diff = max(abs(x - y) for x, y in zip(fa, fb)); mag = max(abs(x) for x in fa)
        nd = sum(1 for x, y in zip(fa, fb) if x != y)
        print(f'first difference at step {i}: tokens {ta}/{tb} pos {pa}/{pb}, {nd} of {len(fa)} values differ, max |diff| {diff:.3e} (max |x| {mag:.3e})')
        break
else:
    print('identical over', n, 'steps')
