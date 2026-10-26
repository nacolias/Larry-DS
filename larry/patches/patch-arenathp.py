#!/usr/bin/env python3
# patch-arenathp.py [TIER_CPP] - larry, STRATA_XE_ARENA_THP=1 (default 0 until measured): the pageable arena chunks
# (posix_memalign, 2 MiB aligned) advised MADV_HUGEPAGE (larry: THP "madvise").  The prefetch's staging copies whole
# experts out of them at ~3.5 GB/s a stager thread whatever the ring, chunk or stager count (abwT33 / T35): on 4 KiB
# pages the hardware prefetchers stop at every page.  (Tried on 2026-10-21, before the hot rings, with no gain - the
# copies then went to DRAM three times.)  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_ARENA_THP" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""            if (posix_memalign(&q, 2u << 20, chunk_size[c]) != 0 || !q) { err = "xe tier: arena allocation failed"; return false; }
            p = (uint8_t*) q;
""", """            if (posix_memalign(&q, 2u << 20, chunk_size[c]) != 0 || !q) { err = "xe tier: arena allocation failed"; return false; }
            p = (uint8_t*) q;
            // larry, STRATA_XE_ARENA_THP: 2 MiB pages for the staging's streaming reads (faulted in when the arena fills)
            static const int arena_thp = env_i("STRATA_XE_ARENA_THP", 0);
            if (arena_thp && madvise(q, chunk_size[c], MADV_HUGEPAGE) != 0)
                std::fprintf(stderr, "xe tier: madvise(MADV_HUGEPAGE) on an arena chunk failed (errno %d)\\n", errno);
""")
if "#include <sys/mman.h>" not in s:
    rep("#include <pthread.h>\n", "#include <pthread.h>\n#include <sys/mman.h>\n#include <cerrno>\n")
open(P, "w").write(s)
print("patched", P)
