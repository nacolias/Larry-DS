#!/usr/bin/env python3
# patch-rewind.py [XOLOTL_DIR] - larry, 2026-10-27: a prompt that diverges from the LIVE conversation's last prompt within
# its last 16 positions (a chat template re-rendering its tail, an edited last line) resumes there instead of reading
# everything again - the live counterpart of patch-parkpool.py's partial unpark.  The window ring at the snapshot P holds
# [P - 144, P) and the ratio-2 compressor state P - 2, P - 1: after restore() a resume at an even s >= P - 16 is exact.
# The snapshot does not survive the re-feed; DSpark's window restarts.  STRATA_SERVE_REWIND=0 = off.  With two resident
# conversation slots (patch-seqs-gen.py) a conversation that an unrelated request used to push into the pool stays live,
# so this is where its re-rendered turn lands (s1a: reusetest's "last words re-rendered" read 2,439 tokens).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/engine/ds41_generate.cpp")
s = open(P).read()
if "patch-rewind.py" in s:
    print("already patched", P)
    sys.exit(0)
old = r'''            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK; park_live)'''
new = r'''            } else if (!unparked && snap_len > 0 && lcp < (size_t) snap_len && (lcp & ~(size_t) 1) >= 64 &&
                       (size_t) snap_len - (lcp & ~(size_t) 1) <= 16 && (lcp & ~(size_t) 1) < ids.size() &&
                       !(std::getenv("STRATA_SERVE_REWIND") && std::getenv("STRATA_SERVE_REWIND")[0] == '0') && dense.restore()) {
                // larry (patch-rewind.py): the last prompt's tail re-rendered - resume at the even point under the
                // divergence (the ring at the snapshot holds the 144 positions before it; restore() put it back)
                const size_t s0 = lcp & ~(size_t) 1;
                std::fprintf(stderr, "serve: rewound the live conversation to %zu (lcp %zu, its snapshot %d)\n", s0, lcp, snap_len);
                hist.resize(s0);
                conv_img = img_trim(conv_img, (int64_t) s0);   // (larry, vision)
                reused = (int) s0;
                snap_len = -1;   // the tail under it is re-fed: no snapshot survives this
#if defined(DS41_HAVE_DSPARK)
                dsp_next = -1;   // DSpark's window holds the old tail
#endif
            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK; park_live)'''
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
