import sys,re
# parse unitrace -d device timing summary: "name", calls, time, pct, avg, min, max
rows=[]
for line in open(sys.argv[1]):
    s=line.strip()
    if not s.startswith('"'): continue
    m=re.match(r'^"(.*)",\s*(\d+),\s*(\d+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*(\d+)\s*$',s)
    if not m: continue
    rows.append((m.group(1),int(m.group(2)),int(m.group(3)),int(m.group(5))))
def short(n):
    n=re.sub(r'\(.*','',n) if not n.startswith('zeCommand') else n
    return n[:90]
tier=re.compile(r'strata::v41::xe|gemm_f16f16f32|zeCommandListAppendMemoryCopy\(H2D\)|run_big|run_small')
tot_t=sum(r[2] for r in rows if tier.search(r[0])); tot_d=sum(r[2] for r in rows if not tier.search(r[0]))
print("tier-ish %.2f s   dense-ish %.2f s"%(tot_t/1e9,tot_d/1e9))
for r in sorted(rows,key=lambda r:-r[2])[:45]:
    print("%7.3f s %6d calls avg %8.1f us  %s %s"%(r[2]/1e9,r[1],r[3]/1e3,'T' if tier.search(r[0]) else 'D',short(r[0])))
