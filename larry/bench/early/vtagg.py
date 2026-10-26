import csv, sys
rows = list(csv.reader(open(sys.argv[1]), delimiter='\t'))[1:]
agg = {}
for r in rows:
    if len(r) < 4 or not r[0]:
        continue
    name = r[0]
    if 'xe::(anonymous namespace)::' in name:
        short = 'xe::' + name.split('xe::(anonymous namespace)::')[1].split('(')[0]
    elif '<&(void ' in name:
        short = 'launch:' + name.split('<&(void ')[1].split('(')[0]
    else:
        short = name.split('(')[0]
    t = float(r[1] or 0); n = int(float(r[3] or 0))
    a = agg.setdefault(short, [0.0, 0]); a[0] += t; a[1] += n
ker = {k: v for k, v in agg.items() if not k.startswith('ze')}
print('kernels: %.3f s in %d instances' % (sum(v[0] for v in ker.values()), sum(v[1] for v in ker.values())))
for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print('%8.3f s %8d  %7.1f us  %s' % (v[0], v[1], 1e6 * v[0] / max(v[1], 1), k[:100]))
