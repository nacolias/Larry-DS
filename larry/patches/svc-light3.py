#!/usr/bin/env python3
# svc-light3.py [ENGINE_SH] - the service: STRATA_XE_CPU_LIGHT 4 -> 3 with BAR staging (streams got cheaper than the
# CPU pool's decode of the light experts): abwT87 / abwT88 fresh 1K 2.28 -> 2.17 s, 2K 3.05 -> 2.99, 3K 3.51 -> 3.41,
# 4.5K 4.13 -> 4.02, 8K 4.74 -> 4.68, 512 / 16K / 31K even.  Backup ds41-xe-engine.sh.pre-light3-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
new = 'export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-3}"'
if new in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-4}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy2(P, P + ".pre-light3-20261025")
s = s.replace(old, "# 3 with BAR staging (was 4): the light experts' streams cost less than the pool's decode of them now - fresh 1K-4.5K\n"
              "# prompts -0.05..-0.1 s, 512 / 16K / 31K even (abwT87, abwT88)\n" + new + "\n")
open(P, "w").write(s)
print("patched", P)
