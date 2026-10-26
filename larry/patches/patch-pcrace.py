#!/usr/bin/env python3
# patch-pcrace.py [TIER_CPP] - larry: card-0 CAT errors (2026-10-25 01:24 and 01:54, reproduced): a keep-cache shrink
# freed card-0 cache slabs that still held cached experts.  The chain: the TP core's KV-room eviction (STRATA_XE_TP_EVICT)
# made hot expert-card experts arena experts; decode ran them on the CPU pool and queued their promotions; an 8K prompt
# from empty (no prefetch, so no pc_drop) let the promotions live on while card 0's cache regrew - taking some of the
# SAME experts; the next decode completed those promotions, which set card_of / dptr to the expert card's slot without
# looking; dense_set still listed them, the next shrink's walk of dense_set's suffix stopped at the first moved entry -
# and freed the slab anyway (11 slabs freed, 294 experts released: STRATA_ADDR_LOG + the fault addresses).
#  (1) seed_dense (seed / regrow) skips an expert a promotion holds (queued, filling or done);
#  (2) a promotion that completes for an expert placed meanwhile leaves it where it is (its slot goes free);
#  (3) shrink_cache_for evicts EVERY cached expert in a slab it frees (a full scan) and drops stale list entries;
#  (4) evict_slabs (the expert cards' KV room) likewise finds the slab's experts anywhere past its evictable start.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-pcrace" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


# (1) seed_dense
rep("""        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        const size_t b = im.bbytes(le.first, le.second);
        if (c.used + (int64_t) b > c.budget) continue;
""", """        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        // larry, patch-pcrace: not an expert a promotion holds (queued / filling / done): its completion would move it
        if (key < im.pc_where.size() && im.pc_where[key] != -1) continue;
        const size_t b = im.bbytes(le.first, le.second);
        if (c.used + (int64_t) b > c.budget) continue;
""")

# (2) promotion completion
rep("""            } else {
                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
                rt_put(im, (size_t) j.key);
""", """            } else if (im.card_of[(size_t) j.key] >= 0) {   // larry, patch-pcrace: placed meanwhile - it stays there
                c.pc_key[(size_t) j.slot] = -1;
                im.pc_where[(size_t) j.key] = -1;
            } else {
                im.card_of[(size_t) j.key] = (int8_t) c.idx;
                im.dptr[(size_t) j.key] = slot;
                rt_put(im, (size_t) j.key);
""")

# (3) shrink_cache_for: every cached expert of the freed slab
rep("""    while (c.free_vram() < need_bytes && !c.slabs.empty()) {
        uint8_t* last = c.slabs.back();
        while (!im.dense_set.empty()) {   // the slab's experts: a suffix of dense_set (slabs fill in seeding order)
            const auto& le = im.dense_set.back();
            const size_t key = (size_t) im.key(le.first, le.second);
            const uint8_t* p = im.dptr[key];
            if (p < last || p >= last + kSlab) break;
            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            rt_put(im, key);
            c.used -= (int64_t) im.bbytes(le.first, le.second);
            --c.n_experts;
            im.dense_set.pop_back();
            ++released;
        }
        local_free(c.q, last);
""", """    while (c.free_vram() < need_bytes && !c.slabs.empty()) {
        uint8_t* last = c.slabs.back();
        // larry, patch-pcrace: EVERY cached expert in the slab goes - not just dense_set's suffix (an entry a late
        // promotion had moved stopped that walk, and the slab was freed under live experts: card-0 CAT errors) - and
        // entries no longer on this card leave the list
        size_t w = 0;
        for (size_t i = 0; i < im.dense_set.size(); ++i) {
            const auto le = im.dense_set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            if (im.card_of[key] != (int8_t) c.idx) {   // stale: on another card / in the arena since
                c.used -= (int64_t) im.bbytes(le.first, le.second);
                --c.n_experts;
                continue;
            }
            const uint8_t* p = im.dptr[key];
            if (p >= last && p < last + kSlab) {
                im.card_of[key] = -1;
                im.dptr[key] = nullptr;
                rt_put(im, key);
                c.used -= (int64_t) im.bbytes(le.first, le.second);
                --c.n_experts;
                ++released;
                continue;
            }
            im.dense_set[w++] = le;
        }
        im.dense_set.resize(w);
        local_free(c.q, last);
""")

# (4) evict_slabs: the slab's experts anywhere from `from` on, moved to the set's end
rep("""        size_t k = set.size();   // the slab's experts: a suffix of set
        while (k > 0) {
            const auto& le = set[k - 1];
            const uint8_t* p = im.dptr[(size_t) im.key(le.first, le.second)];
            if (p < last || p >= last + kSlab) break;
            --k;
        }
        if (k < from) break;     // the slab holds experts that are not evictable
""", """        // larry, patch-pcrace: the slab's experts wherever they are in the set (not only its suffix) - moved to its end
        auto in_slab = [&](const std::pair<int32_t, int32_t>& le) {
            const size_t key = (size_t) im.key(le.first, le.second);
            const uint8_t* p = im.dptr[key];
            return im.card_of[key] == (int8_t) c.idx && p >= last && p < last + kSlab;
        };
        bool fixed_in = false;
        for (size_t i = 0; i < from && i < set.size(); ++i) fixed_in = fixed_in || in_slab(set[i]);
        if (fixed_in) break;     // the slab holds experts that are not evictable
        const size_t k = (size_t) (std::stable_partition(set.begin() + (std::ptrdiff_t) from, set.end(),
                                                         [&](const std::pair<int32_t, int32_t>& le) { return !in_slab(le); }) -
                                   set.begin());
""")
open(P, "w").write(s)
print("patched", P)
