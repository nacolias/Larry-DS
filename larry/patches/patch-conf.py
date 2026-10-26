#!/usr/bin/env python3
# patch-conf.py - ds41-xe-engine.sh: DSpark's confidence cut 0.6 -> 0.3 (the cheaper draft makes wider verify passes
# pay: one-shot essay 54.3 -> 58.4 tok/s, p1 73.0 -> 75.3, 2026-10-21 sweep 0.1-0.75).  Idempotent.
import sys, shutil
p = "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(p).read()
old = 'export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"\n'
new = ('# the draft confidence that ends a pass\'s drafts: 0.3 (was 0.6) since the draft is 4.5 ms - wider passes pay\n'
       '# (one-shot essay 54.3 -> 58.4 tok/s, p1 73.0 -> 75.3; 0.1 / 0.2 / 0.45 / 0.75 lower)\n'
       'export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.3}"\n')
if new in s: print("already"); sys.exit(0)
if s.count(old) != 1: sys.exit("anchor")
shutil.copy(p, "/opt/ai/dsv41/bench/ds41-xe-engine.sh.pre-conf")
open(p, "w").write(s.replace(old, new))
print("patched")
