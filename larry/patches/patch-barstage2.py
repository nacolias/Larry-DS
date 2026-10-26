#!/usr/bin/env python3
# patch-barstage2.py [TIER_CPP] - larry, after patch-barstage.py: what a BAR buffer costs to make is its CPU pages
# (TTM maps 64 KiB a fault; a 256 MiB stream buffer took 31-48 ms, abwT82) - so (1) bar_alloc no longer touches them
# itself: prefetch_begin faults the cards' new flex regions in, a thread a card, after all are made (the faults of one
# buffer object serialize on its lock; the cards' run side by side); (2) a BAR stream buffer of the planned size stays
# through prefetch_begin (it freed and re-made it every batch); (3) STRATA_XE_STAGERS defaults to 1 with BAR staging
# (one thread writes ~21 GB/s through a BAR; four of them stole the link from phase A's own transfers: abwT82 8K 5.77 s
# at 4 / 4.90 at 2 / 5.33 at 1, abwT83 after (1)-(2): 1 stager best or even at every size, 8K 4.94 vs 5.08 at 2).
# STRATA_XE_ALLOC_LOG=1 prints the fault-in times.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-barstage2" in s:
    print("already patched"); sys.exit(0)
if "STRATA_XE_BAR_STAGE" not in s:
    sys.exit("patch-barstage.py first")


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    // the CPU mapping's pages now (TTM maps a 64 KiB run a fault): not inside the first prompt's staging
    for (size_t o = 0; o < sz; o += 65536) (void) *(volatile const uint8_t*) ((const uint8_t*) cpu + o);
    g_bar[(uintptr_t) p] = BarBuf{fd, gc.handle, (uint8_t*) cpu, sz};
    return (uint8_t*) p;
}
""", """    g_bar[(uintptr_t) p] = BarBuf{fd, gc.handle, (uint8_t*) cpu, sz};
    return (uint8_t*) p;   // (its CPU pages fault in at first touch: bar_touch, patch-barstage2)
}
""")
rep("""    const uintptr_t off = (uintptr_t) p - it->first;
    return off < it->second.bytes ? it->second.cpu + off : nullptr;
}
""", """    const uintptr_t off = (uintptr_t) p - it->first;
    return off < it->second.bytes ? it->second.cpu + off : nullptr;
}
/// larry, patch-barstage2: fault a BAR buffer's CPU pages in (TTM maps a 64 KiB run a fault); ms taken
inline double bar_touch(const void* p, size_t bytes) {
    uint8_t* c = bar_cpu(p);
    if (!c) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t o = 0; o < bytes; o += 65536) (void) *(volatile const uint8_t*) (c + o);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
""")
# (1) + (2) in prefetch_begin
rep("""    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        if (c.stream) { c.q.wait(); local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        c.st_layer = -1;
        const int64_t have = (int64_t) (c.cX_cap + c.cOUT_cap) * 4 + (int64_t) c.cS_cap * 8 + (int64_t) c.gscratch_cap +
                             (c.cjobs_cap ? BJ * kMaxJobTok * (F + H) * 4 : 0) + (int64_t) c.cxq_rows * H * 2;
        const int64_t reserve = std::max<int64_t>(0, rb - have) + (int64_t) (margin * kGiB);
        const int64_t fr = c.free_vram() + (int64_t) c.flex_cap;   // the flex region becomes the buffers
""", """    std::vector<XeCard*> touch;   // larry, patch-barstage2: new BAR flex regions, faulted in below (a thread a card)
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        // larry, patch-barstage2: a BAR stream buffer of the planned size stays (a new one costs its CPU pages)
        const size_t sb0 = (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB) & ~(size_t) 255;
        const bool keep_st = c.stream && c.stream_bytes == sb0 && bar_cpu(c.stream);
        if (c.stream && !keep_st) { c.q.wait(); local_free(c.q, c.stream); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        c.st_layer = -1;
        const int64_t have = (int64_t) (c.cX_cap + c.cOUT_cap) * 4 + (int64_t) c.cS_cap * 8 + (int64_t) c.gscratch_cap +
                             (c.cjobs_cap ? BJ * kMaxJobTok * (F + H) * 4 : 0) + (int64_t) c.cxq_rows * H * 2;
        const int64_t reserve = std::max<int64_t>(0, rb - have) + (int64_t) (margin * kGiB);
        const int64_t fr = c.free_vram() + (int64_t) c.flex_cap + (keep_st ? (int64_t) sb0 : 0);   // the flex region becomes the buffers
""")
rep("""            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (!c.flex) { c.pbuf_cap = 0; continue; }
        }
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + cap;
        c.pbuf_cap = cap;
    }
""", """            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (!c.flex) { c.pbuf_cap = 0; continue; }
            if (bar_cpu(c.flex)) touch.push_back(&c);
        }
        c.pbuf[0] = c.flex;
        c.pbuf[1] = c.flex + cap;
        c.pbuf_cap = cap;
    }
    if (!touch.empty()) {   // larry, patch-barstage2: the new BAR regions' CPU pages, the cards side by side
        std::vector<double> ms(touch.size(), 0.0);
        std::vector<std::thread> th;
        for (size_t i = 0; i < touch.size(); ++i) th.emplace_back([&, i] { ms[i] = bar_touch(touch[i]->flex, touch[i]->flex_cap); });
        for (auto& t : th) t.join();
        static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
        if (all)
            for (size_t i = 0; i < touch.size(); ++i)
                std::fprintf(stderr, "xe prefetch: card %d's BAR flex region (%.2f GiB) faulted in: %.0f ms\\n", touch[i]->idx,
                             touch[i]->flex_cap / kGiB, ms[i]);
    }
""")
# (3) two stagers by default with BAR staging
rep("""const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));""",
    """const int ns = std::max(1, env_i("STRATA_XE_STAGERS", bar_stage_on() ? 1 : 4));   // larry, patch-barstage2: 1 with BAR""", 2)
open(P, "w").write(s)
print("patched", P)
