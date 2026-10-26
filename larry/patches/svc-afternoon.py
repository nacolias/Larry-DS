#!/usr/bin/env python3
# svc-afternoon.py [ENGINE_SH] KNOB... - larry: the 2026-10-24 afternoon A/B's winners into ds41-xe-engine.sh, each
# with its comment, before the exec line.  KNOB = depth | lookup | keep | flex.  Backs the file up first
# (.pre-afternoon-20261024).  Idempotent per knob.
import shutil
import sys

args = sys.argv[1:]
P = "/opt/ai/dsv41/ds41-xe-engine.sh"
if args and args[0].startswith("/"):
    P = args.pop(0)
BLOCKS = {
    "depth": ("STRATA_XE_PREFETCH_DEPTH", """# a prompt batch that ends before context position 8192 takes no prefetch: its routed arena experts stream on demand
# and the CPU pool runs the light ones (CPU_LIGHT) - shallow batches are bound by their experts (2026-10-24 A/B: see
# the README)
export STRATA_XE_PREFETCH_DEPTH="${STRATA_XE_PREFETCH_DEPTH-8192}"
"""),
    "lookup": ("STRATA_LOOKUP_TP", """# lookup drafts (STRATA_SERVE_LOOKUP) stay within the TP core's 7-row pass: a wider lookup pass took the old path,
# which pages card 0's dense weights in for every layer (~0.1-0.2 s a pass; 2026-10-24 A/B: see the README)
export STRATA_LOOKUP_TP="${STRATA_LOOKUP_TP-1}"
"""),
    "keep": ("STRATA_SERVE_KEEP_MAX", """# keep-cache up to 8192-token batches (was 4096): the batch gives back only the VRAM it needs from card 0's expert
# cache, and the decode after it regrows the tail instead of re-seeding the whole cache (~0.9 s)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-8192}"
"""),
    "flex": ("STRATA_XE_FLEX_GIB=\"${STRATA_XE_FLEX_GIB-2.5}", None),
}
s = open(P).read()
shutil.copy(P, P + ".pre-afternoon-20261024")
anchor = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"\n'
if s.count(anchor) != 1:
    sys.exit("exec anchor x%d" % s.count(anchor))
for k in args:
    key, text = BLOCKS[k]
    if k == "flex":
        old_r = 'export STRATA_XE_RESERVE_GIB="${STRATA_XE_RESERVE_GIB:-3.5}"'
        new_r = 'export STRATA_XE_RESERVE_GIB="${STRATA_XE_RESERVE_GIB:-5.0}"'
        old_f = 'export STRATA_XE_FLEX_GIB="${STRATA_XE_FLEX_GIB-1}"'
        new_f = 'export STRATA_XE_FLEX_GIB="${STRATA_XE_FLEX_GIB-2.5}"'
        if new_f in s:
            continue
        if s.count(old_r) != 1 or s.count(old_f) != 1:
            sys.exit("flex anchors %d %d" % (s.count(old_r), s.count(old_f)))
        s = s.replace(old_r, new_r).replace(old_f, new_f)
        continue
    if key in s:
        continue
    s = s.replace(anchor, text + anchor)
open(P, "w").write(s)
print("patched", P, args)
