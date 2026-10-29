#!/usr/bin/env python3
# svc-pclend.py [ENGINE_SH] - the service (2026-10-29, decode at depth): STRATA_XE_PC_LEND=1 (patch-pclend.py) - the expert
# cards' free VRAM (STRATA_XE_FLEX_MARGIN_GIB's 1.5 GiB kept for prompts, less STRATA_XE_PC_LEND_KEEP_GIB 0.4) lent to the
# promotion cache between TP passes and given back before a prompt call over 8 tokens, the prefetch or the TP core's KV
# rows.  Backup ds41-xe-engine.sh.pre-pclend-20261029; the file is replaced by a rename.  Idempotent.
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_PC_LEND" in s:
    print("already patched"); sys.exit(0)
old = 'export STRATA_XE_FLEX_MARGIN_GIB="${STRATA_XE_FLEX_MARGIN_GIB-1.5}"\n'
if s.count(old) != 1:
    sys.exit("anchor x%d: %s" % (s.count(old), old[:80]))
s = s.replace(old, old +
              '# between prompts that margin (less 0.4 GiB) is lent to the promotion cache as more slots and given back before a\n'
              '# prompt call over 8 tokens, the prefetch or the TP core\'s KV rows (patch-pclend.py, 2026-10-29: decode at depth)\n'
              'export STRATA_XE_PC_LEND="${STRATA_XE_PC_LEND-1}"\n')
shutil.copy2(P, P + ".pre-pclend-20261029")
tmp = P + ".new"
open(tmp, "w").write(s)
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("patched", P)
