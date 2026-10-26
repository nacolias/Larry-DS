#!/usr/bin/env python3
# patch-hotnta.py [TIER_CPP] - larry, STRATA_XE_HOTCOPY_NTA=1 (diagnostic, default 0): hot_copy's software prefetch of the
# pageable source with the NTA hint (STRATA_XE_HOTCOPY_PF bytes ahead, 1024 when unset).  Each ring chunk's copy pulls
# 1 MiB of source through the L2 into the L3 beside the 1 MiB it writes; at ~26 GB/s of copies the 32 MB L3 turns over
# in ~1 ms - about a chunk's DMA latency in a prompt (STRATA_XE_HOT_PROF ~0.6 ms) - so ring lines can leave the L3 before
# the card reads them (then the DMA reads DRAM and the line was written back).  Non-temporal source lines should not
# displace them.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOTCOPY_NTA" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    static const size_t pf_d = (size_t) std::max(0, env_i("STRATA_XE_HOTCOPY_PF", 0));
""", """    // larry, STRATA_XE_HOTCOPY_NTA (patch-hotnta.py): the same prefetch with the non-temporal hint
    static const bool pf_nta = env_i("STRATA_XE_HOTCOPY_NTA", 0) != 0;
    static const size_t pf_d = (size_t) std::max(0, env_i("STRATA_XE_HOTCOPY_PF", pf_nta ? 1024 : 0));
""")
rep("""            if (i + pf_d < n) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_T0);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_T0);
            }""", """            if (i + pf_d < n && pf_nta) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_NTA);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_NTA);
            } else if (i + pf_d < n) {
                _mm_prefetch((const char*) (src + i + pf_d), _MM_HINT_T0);
                _mm_prefetch((const char*) (src + i + pf_d + 64), _MM_HINT_T0);
            }""")
open(P, "w").write(s)
print("patched", P)
