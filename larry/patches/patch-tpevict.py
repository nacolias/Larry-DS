#!/usr/bin/env python3
# patch-tpevict.py [TIER_CPP] - larry, engine1, STRATA_XE_TP_EVICT=1 (default 0): the expert cards keep their experts in
# the VRAM the TP core's KV replicas will map only at long contexts.  With a 1M-token context the replicas need ~3.1 GiB
# a card (0.78 at 262K); the tier lent all of it to the flex region at load, so the cards held 757 fewer experts and a
# short-context agent replay ran ~8 % slower.  Now only STRATA_XE_TP_DEFER_FLEX_GIB (0.8: what 262K needs) is lent to the
# flex region; the rest of the deferred VRAM holds experts placed as usual, in each card's LAST slabs (whole slabs,
# ceil(rest / 1 GiB)), with a copy in the arena (pageable, after the true arena experts).  When the KV rows need more
# than the card's free VRAM above STRATA_XE_TP_EVICT_KEEP_GIB (1.5: most of the prompt reserve stays), tp_make_room
# gives those slabs back - their experts become arena experts (card_of -1, route tables dirty) - before it shrinks the
# flex region.  Between TP passes only (the TP core asks from ensure_rows).  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_TP_EVICT" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


# ---- Ds41XeImpl: per card its static experts in upload (= slab) order, where the evictable suffix starts
rep("""    std::map<int, size_t> tp_defer;    ///< larry, engine1: by GPU, the TP core's VRAM not mapped yet (the flex holds it)
""", """    std::map<int, size_t> tp_defer;    ///< larry, engine1: by GPU, the TP core's VRAM not mapped yet (the flex holds it)
    /// larry, engine1, STRATA_XE_TP_EVICT: per card its static experts in upload order (slabs fill in that order) and the
    /// index from which they may go back to the arena (whole slabs; each has an arena copy)
    std::vector<std::vector<std::pair<int32_t, int32_t>>> card_set;
    std::vector<size_t> evict_from;
    int64_t evicted_n = 0;
""")

# ---- budgets: in evict mode only the flex's share of the deferred VRAM is held back
rep("""        // larry, engine1: VRAM the TP core will map later (its KV replicas) is lent to the flex region, not to experts
        const auto td = im.tp_defer.find(c.gpu);
        if (td != im.tp_defer.end()) fr -= (int64_t) td->second;
        double gib = fr / kGiB - reserve;
""", """        // larry, engine1: VRAM the TP core will map later (its KV replicas) is lent to the flex region, not to experts
        // (STRATA_XE_TP_EVICT: only STRATA_XE_TP_DEFER_FLEX_GIB of it - the rest holds evictable experts, below)
        const auto td = im.tp_defer.find(c.gpu);
        if (td != im.tp_defer.end()) fr -= tp_evict_on() ? std::min<int64_t>((int64_t) td->second, tp_defer_flex()) : (int64_t) td->second;
        double gib = fr / kGiB - reserve;
""")

# ---- after dealing: each card's evictable suffix (whole slabs replayed as alloc_expert will fill them), added to the
# arena list after the true arena experts
rep("""    if (filled) std::fprintf(stderr, "xe tier: %.2f GiB more experts in the cards' last slabs (allocated anyway)\\n", filled / kGiB);
""", """    if (filled) std::fprintf(stderr, "xe tier: %.2f GiB more experts in the cards' last slabs (allocated anyway)\\n", filled / kGiB);
    // larry, engine1, STRATA_XE_TP_EVICT: the cards' last slabs over the deferred VRAM beyond the flex's share are
    // evictable - their experts get arena copies too (after the arena's own experts: pageable)
    im.card_set.assign(im.cards.size(), {});
    im.evict_from.assign(im.cards.size(), 0);
    std::vector<std::pair<int32_t, int32_t>> evictable;
    for (auto& cp : im.cards) {
        const XeCard& c = *cp;
        const size_t ci = (size_t) c.idx;
        im.card_set[ci] = on[ci];
        im.evict_from[ci] = on[ci].size();
        const auto td = im.tp_defer.find(c.gpu);
        if (!tp_evict_on() || c.dense || td == im.tp_defer.end() || (int64_t) td->second <= tp_defer_flex()) continue;
        const int64_t want = (int64_t) td->second - tp_defer_flex();
        const size_t nslab = (size_t) ((want + (int64_t) kSlab - 1) / (int64_t) kSlab);
        std::vector<size_t> start;   // the index of each slab's first expert (alloc_expert's bump replayed)
        int64_t u = -1;
        for (size_t i = 0; i < on[ci].size(); ++i) {
            const int64_t b = (int64_t) im.bbytes(on[ci][i].first, on[ci][i].second);
            if (u < 0 || u + b > (int64_t) kSlab) { start.push_back(i); u = 0; }
            u += (b + 255) & ~(int64_t) 255;
        }
        if (start.size() <= nslab) continue;   // keep at least one slab
        im.evict_from[ci] = start[start.size() - nslab];
        for (size_t i = im.evict_from[ci]; i < on[ci].size(); ++i) evictable.push_back(on[ci][i]);
    }
""")
rep("""    for (const auto& le : rest) {
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        if (abytes + b > cap) { ++im.file_n; continue; }
""", """    const size_t n_rest = rest.size();
    rest.insert(rest.end(), evictable.begin(), evictable.end());   // STRATA_XE_TP_EVICT: their copies, last
    for (size_t ri = 0; ri < rest.size(); ++ri) {
        const auto& le = rest[ri];
        const int64_t b = (int64_t) im.bbytes(le.first, le.second);
        if (abytes + b > cap) { if (ri < n_rest) ++im.file_n; continue; }
""")

# ---- after the arena's pointers: a card whose evictable experts lack copies (the RAM cap) evicts nothing
rep("""    std::fprintf(stderr, "xe tier: arena %.2f GiB in %zu chunks (%.2f GiB pinned), %lld experts; file tier %lld\\n", abytes / kGiB,
                 nchunks, pinned / kGiB, (long long) im.arena_n, (long long) im.file_n);
""", """    std::fprintf(stderr, "xe tier: arena %.2f GiB in %zu chunks (%.2f GiB pinned), %lld experts; file tier %lld\\n", abytes / kGiB,
                 nchunks, pinned / kGiB, (long long) im.arena_n, (long long) im.file_n);
    for (auto& cp : im.cards) {   // STRATA_XE_TP_EVICT
        const size_t ci = (size_t) cp->idx;
        if (ci >= im.card_set.size() || im.evict_from[ci] >= im.card_set[ci].size()) continue;
        bool all = true;
        int64_t eb = 0;
        for (size_t i = im.evict_from[ci]; i < im.card_set[ci].size(); ++i) {
            const auto& le = im.card_set[ci][i];
            all = all && im.aptr[(size_t) im.key(le.first, le.second)];
            eb += (int64_t) im.bbytes(le.first, le.second);
        }
        if (!all) {
            std::fprintf(stderr, "xe tier: card %d: its evictable experts have no arena copies (RAM cap) - none evictable\\n", cp->idx);
            im.evict_from[ci] = im.card_set[ci].size();
            continue;
        }
        std::fprintf(stderr, "xe tier: card %d: %zu experts (%.2f GiB, its last slabs) go back to the arena when the TP core's KV "
                     "rows need the room (STRATA_XE_TP_EVICT)\\n", cp->idx, im.card_set[ci].size() - im.evict_from[ci], eb / kGiB);
    }
""")

# ---- the flex region: only its share of the deferred VRAM
rep("""            const auto td = im.tp_defer.find(c.gpu);
            const int64_t extra = td == im.tp_defer.end() ? 0 : (int64_t) td->second;
""", """            const auto td = im.tp_defer.find(c.gpu);
            int64_t extra = td == im.tp_defer.end() ? 0 : (int64_t) td->second;
            if (tp_evict_on()) extra = std::min<int64_t>(extra, tp_defer_flex());   // STRATA_XE_TP_EVICT: the rest holds experts
""")

# ---- helpers (before build_arena_ranked, with the hot-ring forward declarations)
rep("""namespace {
bool hot_on();
size_t hot_chunk();
int hot_slots();
}  // namespace
""", """namespace {
bool hot_on();
size_t hot_chunk();
int hot_slots();
void rt_put(Ds41XeImpl& im, size_t key);
void rep_drop(Ds41XeImpl& im);
/// larry, engine1, STRATA_XE_TP_EVICT (default 0): see tp_make_room / build_arena_ranked
bool tp_evict_on() { static const bool on = env_i("STRATA_XE_TP_EVICT", 0) != 0; return on; }
int64_t tp_defer_flex() {
    static const int64_t b = (int64_t) (env_d("STRATA_XE_TP_DEFER_FLEX_GIB", 0.8) * 1073741824.0);
    return b;
}
/// the card's evictable last slabs back to the arena until `need` bytes are free; their experts become arena experts.
/// Between TP passes only: no kernel reads them (the card's queues are idle; promotions write flex slots).
int64_t evict_slabs(Ds41XeImpl& im, XeCard& c, int64_t need) {
    const size_t ci = (size_t) c.idx;
    if (ci >= im.card_set.size()) return 0;
    auto& set = im.card_set[ci];
    const size_t from = im.evict_from[ci];
    if (set.size() <= from || c.slabs.size() < 2) return 0;
    const double t0 = now_ms();
    rep_drop(im);   // the dense card's copies of expert-card experts first (route tables: those entries change too)
    c.q.wait();
    int64_t n = 0, bytes = 0;
    while (c.free_vram() < need && c.slabs.size() > 1 && set.size() > from) {
        uint8_t* last = c.slabs.back();
        size_t k = set.size();   // the slab's experts: a suffix of set
        while (k > 0) {
            const auto& le = set[k - 1];
            const uint8_t* p = im.dptr[(size_t) im.key(le.first, le.second)];
            if (p < last || p >= last + kSlab) break;
            --k;
        }
        if (k < from) break;     // the slab holds experts without arena copies
        for (size_t i = k; i < set.size(); ++i) {
            const auto& le = set[i];
            const size_t key = (size_t) im.key(le.first, le.second);
            im.card_of[key] = -1;
            im.dptr[key] = nullptr;
            rt_put(im, key);
            c.used -= (int64_t) im.bbytes(le.first, le.second);
            bytes += (int64_t) im.bbytes(le.first, le.second);
            --c.n_experts;
            ++n;
        }
        set.resize(k);
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
    }
    if (!n) return 0;
    // the driver may report the freed pages a little later
    for (int i = 0; i < 100 && c.free_vram() < need; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    im.resident_n = 0;
    for (auto& cp : im.cards) im.resident_n += cp->n_experts;
    im.evicted_n += n;
    std::fprintf(stderr, "xe tier: card %d gave %lld experts (%.2f GiB) back to the arena for the TP core's KV rows (%.0f ms; "
                 "%.2f GiB free)\\n", c.idx, (long long) n, bytes / 1073741824.0, now_ms() - t0, c.free_vram() / 1073741824.0);
    return n;
}
}  // namespace
""")

# ---- tp_make_room: the evictable slabs before the flex region (most of the prompt reserve kept)
rep("""    if (fr < 0) return true;
    int64_t give = (int64_t) bytes + keep - fr;
    if (give <= 0) return true;
    if (!c.flex || c.pbuf_cap) return fr > (int64_t) bytes;   // no flex region to give back (a prompt holds it)
""", """    if (fr < 0) return true;
    int64_t give = (int64_t) bytes + keep - fr;
    // larry, STRATA_XE_TP_EVICT: the free VRAM down to STRATA_XE_TP_EVICT_KEEP_GIB (most of the prompt reserve), then
    // the card's evictable slabs, the flex region last
    if (tp_evict_on() && !c.pbuf_cap) {
        const int64_t ekeep = (int64_t) (env_d("STRATA_XE_TP_EVICT_KEEP_GIB", 1.5) * kGiB);
        if ((int64_t) bytes + ekeep > fr && evict_slabs(im, c, (int64_t) bytes + ekeep) > 0) give = (int64_t) bytes + keep - c.free_vram();
    }
    if (give <= 0) return true;
    if (!c.flex || c.pbuf_cap) return c.free_vram() > (int64_t) bytes;   // no flex region to give back (a prompt holds it)
""")
open(P, "w").write(s)
print("patched", P)
