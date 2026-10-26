#!/usr/bin/env python3
# patch-ovlring.py [TIER_CPP] - larry: STRATA_XE_OVERLAP's helper streams a round's PAGEABLE experts through the card's
# hot ring (STRATA_XE_HOTRING; a DMA a chunk as it is copied, read from the L3) instead of memcpy'ing whole experts into
# the bounce first (~13 vs ~21.7 GB/s in bench/dqb/stage_ring) - with OVERLAP=1 the on-demand turns (1.5-2.2K tokens,
# no prefetch) got 0.05-0.3 s slower (abwT18).  No bounce-sized rounds then.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-ovlring" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        for (int32_t e : streamed[(size_t) c.idx]) if (!im.aptr[(size_t) im.key(l, e)]) overlap = false;   // file tier: old path
""", """        for (int32_t e : streamed[(size_t) c.idx]) if (!im.aptr[(size_t) im.key(l, e)]) overlap = false;   // file tier: old path
        // larry, patch-ovlring: the helper puts pageable experts through the card's hot ring (no bounce-sized rounds)
        if (overlap && hot_on() && !c.hring) c.hring = sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx);
        const bool ring_ovl = overlap && hot_on() && c.hring;
""")
rep("""            const size_t pg = im.apin[(size_t) im.key(l, e)] ? 0 : b;
""", """            const size_t pg = im.apin[(size_t) im.key(l, e)] || ring_ovl ? 0 : b;
""")
rep("""                    const uint8_t* src = im.aptr[key];
                    if (!im.apin[key]) {
                        std::memcpy(c.bounce + bo, src, bytes);
""", """                    const uint8_t* src = im.aptr[key];
                    if (!im.apin[key] && ring_ovl) {   // larry, patch-ovlring
                        oev = hot_put(c.oq, c.stream + soff[(size_t) e], src, bytes, false, c.hring, c.hring_ev, c.hring_next);
                        ++pcie;
                        ++g_bp.streamed;
                        continue;
                    }
                    if (!im.apin[key]) {
                        std::memcpy(c.bounce + bo, src, bytes);
""")
open(P, "w").write(s)
print("patched", P)
