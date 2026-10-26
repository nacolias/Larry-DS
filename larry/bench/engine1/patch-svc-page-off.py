# patch-svc-page-off.py - ds41-xe-engine.sh: engine1's pager OFF again (long-prompt needles missed with it on)
import sys
p = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(p).read()
old = """# one-shot).  The p2k row that sometimes goes wrong (row 111) does so with the pager off too - a chaotic row, not a
# race (2026-10-23, engine1/DESIGN.md).  0 = card 0 keeps the copies
export STRATA_TP_PAGE="${STRATA_TP_PAGE-1}"
"""
new = """# one-shot).  The p2k row that sometimes goes wrong (row 111) does so with the pager off too - a chaotic row, not a
# race (2026-10-23, engine1/DESIGN.md).  OFF again: in the service the 8K / 16K / 31K needles were missed with it on
# (2026-10-23 05:25, tsvc2) - not understood yet.  0 = card 0 keeps the copies
export STRATA_TP_PAGE="${STRATA_TP_PAGE-0}"
"""
if s.count(old) != 1: sys.exit('pattern not found once')
open(p + '.page-on', 'w').write(s)
open(p, 'w').write(s.replace(old, new))
print('ok')
