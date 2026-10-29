#!/usr/bin/env python3
# patch-pclend.py [XOLOTL_DIR] - larry, 2026-10-29 (decode at depth): STRATA_XE_PC_LEND=1 - the expert cards' free VRAM
# lent to the promotion cache between TP passes.  The cache (STRATA_XE_FLEX_GIB's flex regions, ~380-415 slots) is the
# decode-at-depth lever: an agent session's replies at 40-155K ran 15.4 CPU-pool experts a pass, ~0.5 ms each on the
# critical path; replayed on a deep agent session's log (bench/loop/pcsim2.py) 380 slots leave 10.9 a pass, 600 6.9, 800
# 5.8, while the cards keep ~1.3-1.5 GiB free between prompts for the prompt path.  So: after STRATA_XE_PC_LEND_AFTER (4)
# TP passes with no prompt call, each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB (0.4) becomes a second
# slot region (pc_mem2: its slots after the flex region's, one LRU over both); it goes back - its experts to the CPU
# pool, the region freed, the driver's free VRAM waited for - before anything sizes or allocates from the free VRAM: a
# prompt call over kMaxJobTok tokens (run_big), the prefetch (pc_drop), the TP core's KV rows (tp_make_room_multi,
# tp_room_ahead), close.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
T = os.path.join(X, "tools/v41/engine/xe/ds41_xe_tier.cpp")
s = open(T).read()
if "patch-pclend.py" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, what, count=1):
    global s
    if s.count(old) != count:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    s = s.replace(old, new)


rep("""    uint8_t* pc_tmp = nullptr;      ///< larry, engine1 (pc_devrepack): one slot's bytes the card repacks a promoted blob into
""", """    uint8_t* pc_tmp = nullptr;      ///< larry, engine1 (pc_devrepack): one slot's bytes the card repacks a promoted blob into
    /// larry (patch-pclend.py, STRATA_XE_PC_LEND): free VRAM lent to the cache between TP passes - slots pc_n1 .. pc_n
    /// live there (pc_n1 .. in the flex region before them); null when nothing is lent
    uint8_t* pc_mem2 = nullptr;
    size_t pc_cap2 = 0;
    int pc_n1 = 0;
""", "card fields")
rep("""    bool pc_tried = false;             ///< pc_setup ran since the last drop
""", """    bool pc_tried = false;             ///< pc_setup ran since the last drop
    // larry (patch-pclend.py): the lent slots - TP passes since they went back, the next pass a lend may be tried at,
    // counts and host time (lend: the allocations; unlend: the waits, frees and the driver's free VRAM)
    int64_t pc_unlend_pass = 0, pc_lend_next = 0, pc_lends = 0, pc_unlends = 0, pc_lent_max = 0;
    double pc_lend_ms = 0, pc_unlend_ms = 0;
""", "impl fields")
rep("""        c.pc_mem = c.flex;
        c.pc_n = n;
""", """        c.pc_mem = c.flex;
        c.pc_n = n;
        c.pc_n1 = n;   // larry (patch-pclend.py)
""", "pc_setup")
rep("""        uint8_t* slot = c.pc_mem + (size_t) j.slot * im.pc_slot;
        if (j.stage == 0 && j.filled.load(std::memory_order_acquire)) {""",
    """        uint8_t* slot = pc_addr(im, c, j.slot);   // larry (patch-pclend.py)
        if (j.stage == 0 && j.filled.load(std::memory_order_acquire)) {""", "pc_poll")
rep("""/// STRATA_XE_PROMOTE_CHECK=1 (debug): a published slot against the arena blob repacked afresh on the same card
""", """/// larry (patch-pclend.py): slot sl's bytes - the flex region's slots first, then the lent region's
inline uint8_t* pc_addr(const Ds41XeImpl& im, const XeCard& c, int sl) {
    return sl < c.pc_n1 ? c.pc_mem + (size_t) sl * im.pc_slot : c.pc_mem2 + (size_t) (sl - c.pc_n1) * im.pc_slot;
}

/// STRATA_XE_PROMOTE_CHECK=1 (debug): a published slot against the arena blob repacked afresh on the same card
""", "pc_addr")
# pc_drop frees a lent region too (its slots are unpublished with the rest)
rep("""        c.pc_mem = nullptr;   // the flex region (prefetch buffers next)
        c.pc_tmp = nullptr;
        c.pc_n = 0;
""", """        if (c.pc_mem2) {   // larry (patch-pclend.py): the lent region back with the rest
            local_free(c.q, c.pc_mem2);
            c.pc_mem2 = nullptr;
            c.pc_cap2 = 0;
            ++im.pc_unlends;
        }
        c.pc_n1 = 0;
        c.pc_mem = nullptr;   // the flex region (prefetch buffers next)
        c.pc_tmp = nullptr;
        c.pc_n = 0;
""", "pc_drop")
rep("""    std::fill(im.pc_where.begin(), im.pc_where.end(), -1);
    im.pc_ready = false;
}
""", """    std::fill(im.pc_where.begin(), im.pc_where.end(), -1);
    im.pc_ready = false;
    im.pc_unlend_pass = im.tp_pass;   // (larry, patch-pclend.py)
}

/// larry (patch-pclend.py, STRATA_XE_PC_LEND): each expert card's free VRAM less STRATA_XE_PC_LEND_KEEP_GIB becomes more
/// slots (pc_mem2, after the flex region's) - between TP passes only, nothing in flight reads a slot of it yet
void pc_lend(Ds41XeImpl& im) {
    static const int64_t keep = (int64_t) (env_d("STRATA_XE_PC_LEND_KEEP_GIB", 0.4) * kGiB);
    static const int max_n = std::max(0, env_i("STRATA_XE_PC_LEND_MAX", 4096));
    const double t0 = now_ms();
    int added = 0;
    std::string per;
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense || !c.pc_mem || c.pc_mem2 || c.pbuf_cap) continue;
        const int64_t fr = c.free_vram();
        if (fr < 0) continue;
        const int n = (int) std::min<int64_t>(max_n, std::max<int64_t>(0, fr - keep) / (int64_t) im.pc_slot);
        if (n < 8 || c.pc_n + n > 65535) continue;
        uint8_t* p = local_alloc(c.q, (size_t) n * im.pc_slot);
        if (!p) continue;
        c.pc_mem2 = p;
        c.pc_cap2 = (size_t) n * im.pc_slot;
        c.pc_n1 = c.pc_n;
        c.pc_n += n;
        c.pc_key.resize((size_t) c.pc_n, -1);
        c.pc_tick.resize((size_t) c.pc_n, 0);
        added += n;
        per += " c" + std::to_string(c.idx) + " +" + std::to_string(n);
    }
    // a card with no room is tried again after another STRATA_XE_PC_LEND_AFTER passes, not every pass
    im.pc_lend_next = im.tp_pass + std::max(1, env_i("STRATA_XE_PC_LEND_AFTER", 4));
    if (!added) return;
    ++im.pc_lends;
    im.pc_lend_ms += now_ms() - t0;
    int64_t lent = 0;
    for (auto& cp : im.cards) if (cp->pc_mem2) lent += cp->pc_n - cp->pc_n1;
    im.pc_lent_max = std::max(im.pc_lent_max, lent);
    static int logged = 0;
    if (logged++ < 3 || env_i("STRATA_XE_PROF", 0))
        std::fprintf(stderr, "xe tier: promotion cache: +%d slots (%.2f GiB) of the expert cards' free VRAM lent (%s; %.1f ms)\\n",
                     added, added * (double) im.pc_slot / kGiB, per.c_str() + 1, now_ms() - t0);
}

/// larry (patch-pclend.py): the lent slots back before anything sizes or allocates from the free VRAM: their promotions
/// in flight dropped, their experts back to the CPU pool, the regions freed, the driver's free VRAM waited for
void pc_unlend(Ds41XeImpl& im) {
    bool any = false;
    for (auto& cp : im.cards) any = any || cp->pc_mem2;
    if (!any) return;
    const double t0 = now_ms();
    for (;;) {   // the helper's fills in flight (a job of a lent slot may be among them)
        bool busy = false;
        for (auto& j : im.pc_jobs) if (j.stage == 0 && !j.filled.load(std::memory_order_acquire)) busy = true;
        if (!busy) break;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.pc_mem2) continue;
        c.q.wait();
        if ((size_t) c.idx < im.pc_q.size() && im.pc_q[(size_t) c.idx]) im.pc_q[(size_t) c.idx]->wait();   // the copies
        for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {   // promotions into a lent slot: dropped
            if (it->card == c.idx && it->slot >= c.pc_n1) {
                if (it->bounce >= 0) im.pc_bfree.push_back(it->bounce);
                im.pc_where[(size_t) it->key] = -1;
                c.pc_busy = std::max(0, c.pc_busy - 1);
                it = im.pc_jobs.erase(it);
            } else {
                ++it;
            }
        }
        for (int sl = c.pc_n1; sl < c.pc_n; ++sl) {
            const int64_t k = c.pc_key[(size_t) sl];
            if (k >= 0) {
                im.card_of[(size_t) k] = -1;
                im.dptr[(size_t) k] = nullptr;
                rt_put(im, (size_t) k);
                im.pc_where[(size_t) k] = -1;
            }
        }
        c.pc_key.resize((size_t) c.pc_n1);
        c.pc_tick.resize((size_t) c.pc_n1);
        c.pc_n = c.pc_n1;
        const int64_t fr0 = c.free_vram(), freed = (int64_t) c.pc_cap2;
        local_free(c.q, c.pc_mem2);
        c.pc_mem2 = nullptr;
        c.pc_cap2 = 0;
        // the xe driver shows freed VRAM a little later (as evict_slabs waits for its slabs)
        for (int i = 0, same = 0; fr0 >= 0 && i < 50 && c.free_vram() < fr0 + freed * 9 / 10 && same < 5; ++i) {
            const int64_t f0 = c.free_vram();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            same = c.free_vram() == f0 ? same + 1 : 0;
        }
    }
    ++im.pc_unlends;
    im.pc_unlend_ms += now_ms() - t0;
    im.pc_unlend_pass = im.tp_pass;
}
""", "pc_lend / pc_unlend")
# lend between TP passes
rep("""    // early bounces no card took (every card at its in-flight cap): given back once filled (the helper is done with them)
    for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {
        if (it->card < 0 && it->filled.load(std::memory_order_acquire)) {""",
    """    // larry (patch-pclend.py, STRATA_XE_PC_LEND): the free VRAM lent as more slots, STRATA_XE_PC_LEND_AFTER passes after
    // the last prompt call (no prefetch, no regrow pending)
    static const bool lend_on = env_i("STRATA_XE_PC_LEND", 0) != 0;
    static const int lend_after = std::max(1, env_i("STRATA_XE_PC_LEND_AFTER", 4));
    if (lend_on && im.pc_ready && !im.pf_on && !im.rg_pending && im.tp_pass - im.pc_unlend_pass >= lend_after &&
        im.tp_pass >= im.pc_lend_next)
        pc_lend(im);
    // early bounces no card took (every card at its in-flight cap): given back once filled (the helper is done with them)
    for (auto it = im.pc_jobs.begin(); it != im.pc_jobs.end();) {
        if (it->card < 0 && it->filled.load(std::memory_order_acquire)) {""", "tp_between_passes")
# give it back first: prompt calls over kMaxJobTok tokens, the TP core's KV rows
rep("""bool Ds41XeTier::run_chunk(int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    try {
        if (n > kMaxJobTok) disarm(*im_);""", """bool Ds41XeTier::run_chunk(int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    try {
        if (n > kMaxJobTok) pc_unlend(*im_);   // larry (patch-pclend.py)
        if (n > kMaxJobTok) disarm(*im_);""", "run_chunk")
rep("""bool Ds41XeTier::run_chunk_dev(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    try {
        if (n > kMaxJobTok) disarm(*im_);""", """bool Ds41XeTier::run_chunk_dev(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    try {
        if (n > kMaxJobTok) pc_unlend(*im_);   // larry (patch-pclend.py)
        if (n > kMaxJobTok) disarm(*im_);""", "run_chunk_dev")
rep("""    if (im.run_thr.joinable()) return false;   // one call in flight
    im.run_ok = true;""", """    if (im.run_thr.joinable()) return false;   // one call in flight
    if (n > kMaxJobTok) pc_unlend(im);   // larry (patch-pclend.py): here, before the call's thread
    im.run_ok = true;""", "run_chunk_dev_async")
rep("""bool Ds41XeTier::tp_make_room_multi(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
""", """bool Ds41XeTier::tp_make_room_multi(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    pc_unlend(im);   // larry (patch-pclend.py): the lent VRAM first - else the free VRAM below looks short and slabs go
""", "tp_make_room_multi")
rep("""void Ds41XeTier::tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
""", """void Ds41XeTier::tp_room_ahead(const std::vector<std::pair<int, size_t>>& reqs) {
    Ds41XeImpl& im = *im_;
    pc_unlend(im);   // larry (patch-pclend.py)
""", "tp_room_ahead")
rep("""    if (im.pc_fill_n > 0)
        std::fprintf(stderr, "xe tier: promotion repacks: %lld, %.1f ms each, %.1f ms queued before a helper took it\\n", (long long) im.pc_fill_n.load(),""",
    """    if (im.pc_lends || im.pc_unlends)   // larry (patch-pclend.py)
        std::fprintf(stderr, "xe tier: promotion cache: free VRAM lent %lld times (at most %lld slots, %.1f ms each), given back %lld "
                     "times (%.1f ms each)\\n", (long long) im.pc_lends, (long long) im.pc_lent_max,
                     im.pc_lend_ms / std::max<int64_t>(1, im.pc_lends), (long long) im.pc_unlends,
                     im.pc_unlend_ms / std::max<int64_t>(1, im.pc_unlends));
    if (im.pc_fill_n > 0)
        std::fprintf(stderr, "xe tier: promotion repacks: %lld, %.1f ms each, %.1f ms queued before a helper took it\\n", (long long) im.pc_fill_n.load(),""",
    "close stats")
open(T, "w").write(s)
print("patched", T)
