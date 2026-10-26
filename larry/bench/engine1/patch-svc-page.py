# patch-svc-page.py - ds41-xe-engine.sh: engine1's pager on (STRATA_TP_PAGE default 1)
import sys
p = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(p).read()
old = """# engine1's pager (card 0's copies of the TP cards' dense weights paged in for prompts only: DSpark +10 %, 95K +13 %
# one-shot) stays OFF until the prompt path's latent race it exposes is fixed - a 3 ms busy kernel on the dense queue
# per layer, no pager at all, makes one prompt row of p2k go wrong too (2026-10-23, engine1/DESIGN.md)
export STRATA_TP_PAGE="${STRATA_TP_PAGE-0}"
"""
new = """# engine1's pager: card 0's copies of the dense weights the TP cards hold (q_b, wo_a, wo_b, shared expert, Engram
# projection: 4.9 GiB) paged in per layer for prompts only; card 0 takes ~360 more experts (DSpark +10 %, 95K +13 %
# one-shot).  The p2k row that sometimes goes wrong (row 111) does so with the pager off too - a chaotic row, not a
# race (2026-10-23, engine1/DESIGN.md).  0 = card 0 keeps the copies
export STRATA_TP_PAGE="${STRATA_TP_PAGE-1}"
"""
if s.count(old) != 1: sys.exit('pattern not found once')
open(p + '.pre-page', 'w').write(s)
open(p, 'w').write(s.replace(old, new))
print('ok')
