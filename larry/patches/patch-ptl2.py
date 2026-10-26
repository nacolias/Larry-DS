#!/usr/bin/env python3
# patch-ptl2.py [TIER_CPP] - larry, STRATA_GPU_PTL fix (after patch-ptl.py): the first form's rings stopped for good a
# few hundred records into a prompt (abwT61: every GPU ring stuck at a flag that never flipped; the dump then waited its
# 3 s every batch).  The monitor cleared each flag it saw - a CPU write into a cache line the GPU writes the next flags
# into: the CPU's dirty line can overwrite the card's (no-snoop) write.  Now the card writes the record's sequence
# number into a flag of its own 64-byte line and the host never writes the flags; a flag still missing is passed once
# a later one of the ring is seen (its time is then the later one's, negative = inferred); the dump waits 200 ms at most.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-ptl2" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""struct PtlRing {
    static constexpr uint32_t kCap = 1u << 16;
    uint32_t* flag = nullptr;   // pinned host memory; nullptr: the host ring
""", """struct PtlRing {   // larry, patch-ptl2: a flag a 64-byte line, written only by the card (the record's sequence number)
    static constexpr uint32_t kCap = 1u << 14, kStride = 16;
    uint32_t* flag = nullptr;   // pinned host memory, kCap * kStride; nullptr: the host ring
""")
rep("""            volatile uint32_t* f = r->flag;
            uint32_t sn = r->seen.load(std::memory_order_relaxed);
            const uint32_t is = r->issued.load(std::memory_order_acquire);
            while (sn < is && f[sn % PtlRing::kCap]) {
                r->rec[sn % PtlRing::kCap].td = now_ms();
                f[sn % PtlRing::kCap] = 0;
                ++sn;
                any = true;
            }
            r->seen.store(sn, std::memory_order_release);
""", """            volatile const uint32_t* f = r->flag;
            uint32_t sn = r->seen.load(std::memory_order_relaxed);
            const uint32_t is = r->issued.load(std::memory_order_acquire);
            auto got = [&](uint32_t i) { return f[(size_t) (i % PtlRing::kCap) * PtlRing::kStride] == i + 1; };
            while (sn < is) {
                if (got(sn)) {
                    r->rec[sn % PtlRing::kCap].td = now_ms();
                } else {   // a later record of this in-order queue seen: this one ran before it (time inferred)
                    uint32_t j = sn + 1;
                    while (j < is && j < sn + 256 && !got(j)) ++j;
                    if (j >= is || j >= sn + 256) break;
                    r->rec[sn % PtlRing::kCap].td = -now_ms();
                }
                ++sn;
                any = true;
            }
            r->seen.store(sn, std::memory_order_release);
""")
rep("""        r->flag = sycl::malloc_host<uint32_t>(PtlRing::kCap, ctx);
        if (!r->flag) { delete r; return nullptr; }
        for (uint32_t k = 0; k < PtlRing::kCap; ++k) r->flag[k] = 0;
""", """        r->flag = sycl::malloc_host<uint32_t>((size_t) PtlRing::kCap * PtlRing::kStride, ctx);
        if (!r->flag) { delete r; return nullptr; }
        std::memset(r->flag, 0, (size_t) PtlRing::kCap * PtlRing::kStride * sizeof(uint32_t));
""")
rep("""    if (r->flag) {
        volatile uint32_t* f = r->flag + k;
        q.single_task([=] { *f = 1u; });
    } else {
""", """    if (r->flag) {
        volatile uint32_t* f = r->flag + (size_t) k * PtlRing::kStride;
        const uint32_t seq = i + 1;
        q.single_task([=] { *f = seq; });
    } else {
""")
rep("""        while (r && r->seen.load(std::memory_order_acquire) < r->issued.load(std::memory_order_acquire) && now_ms() - t0 < 3000)
""", """        while (r && r->seen.load(std::memory_order_acquire) < r->issued.load(std::memory_order_acquire) && now_ms() - t0 < 200)
""")
rep("""            std::fprintf(f, "%d\\t%d\\t%d\\t%d\\t%.3f\\t%.3f\\n", ring, e.tag, e.l, e.a, e.th, e.td);
        }
        r->done = sn;
""", """            std::fprintf(f, "%d\\t%d\\t%d\\t%d\\t%.3f\\t%.3f\\n", ring, e.tag, e.l, e.a, e.th, e.td);
        }
        r->done = sn;
        const uint32_t is = r->issued.load(std::memory_order_acquire);
        if (f && is > sn) std::fprintf(f, "# ring %d: %u records not seen\\n", ring, is - sn);
""")
open(P, "w").write(s)
print("patched", P)
