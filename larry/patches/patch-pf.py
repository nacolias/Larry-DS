# patch-pf.py - the xe tier's prompt prefetch (STRATA_XE_PREFETCH): a large layer-major batch streams each layer's
# arena experts onto the expert cards ahead of that layer's run_big, on a second (copy) queue per card, while the dense
# card works - run_big then reads them in place instead of streaming on its own critical path.
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

hp = root + 'xe/ds41_xe_tier.hpp'
if 'prefetch_begin' not in open(hp).read():
    patch(hp, [('''    void release_chunk() {}
''', '''    void release_chunk() {}
    /// larry, STRATA_XE_PREFETCH (default 1): a layer-major prompt batch whose layers first..last each run one
    /// run_chunk_dev of >= STRATA_XE_PREFETCH_MIN (2048) tokens declares them here; the tier then streams every arena
    /// expert of layer l + 1 onto the expert cards (a copy queue per card; pageable experts staged into pinned memory
    /// by a helper thread first) while layer l computes and the dense card runs layer l + 1, and run_chunk_dev(l + 1)
    /// reads them in place.  prefetch_end() after the batch (frees the cards' buffers).  Returns false = off.
    bool prefetch_begin(int64_t first, int64_t last);
    void prefetch_end();
''')])

cp = root + 'xe/ds41_xe_tier.cpp'
s = open(cp).read()
if 'PfSlot' not in s:
    patch(cp, [
    # ---- per card: the copy queue and the two prefetch buffers
    ('''    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
''', '''    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
    // prompt prefetch (Ds41XeTier::prefetch_begin): a copy queue and two layer buffers (ping-pong by layer parity)
    sycl::queue pq;
    uint8_t* pbuf[2] = {nullptr, nullptr};
    size_t pbuf_cap = 0;
    sycl::event pev[2];             ///< the last prefetch copy into pbuf[s]
'''),
    ('''        c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
''', '''        c->q = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
        c->pq = sycl::queue(im.ctx, c->dev, sycl::property::queue::in_order());
'''),
    # ---- the tier's prefetch state
    ('''    sycl::event hsum_ev;               ///< the copy of hsum (the CPU part) into the dense card's slot
''', '''    sycl::event hsum_ev;               ///< the copy of hsum (the CPU part) into the dense card's slot
    // prompt prefetch (Ds41XeTier::prefetch_begin)
    std::vector<uint8_t> apin;         ///< per (layer, expert): its arena copy is pinned (DMA-able as it is)
    struct PfSlot {
        int64_t layer = -1;            ///< the layer it holds or is filling (-1: none)
        bool enqueued = false;         ///< its copies are on the cards' copy queues
        std::vector<int8_t> card;      ///< [n_experts]: the card the expert goes to, -1 = not prefetched
        std::vector<size_t> off;       ///< [n_experts]: offset in that card's pbuf
        std::vector<size_t> soff;      ///< [n_experts]: offset in stage, SIZE_MAX = straight from the pinned arena
        uint8_t* stage = nullptr;      ///< pinned host staging for the layer's pageable experts
        int staged = 0;                ///< 0 nothing to stage / done, 1 requested (stager thread), under pf_mu
        int64_t n = 0, bytes = 0;      ///< experts / bytes planned
    };
    PfSlot pf[2];
    size_t pf_stage_cap = 0;           ///< bytes per slot's stage
    bool pf_on = false;
    int64_t pf_last = -1;
    int64_t pf_used = 0, pf_planned = 0, pf_bytes = 0;   ///< over a batch (the prof line)
    std::thread pf_thr;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    bool pf_stop = false;
'''),
    # ---- the arena: remember which experts are pinned; the stage buffers come out of the pinned budget
    ('''    const int64_t pin_cap = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
    int64_t pinned = 0;
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        if (pinned + (int64_t) chunk_size[c] <= pin_cap) {
            p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
            if (p) pinned += (int64_t) chunk_size[c];
        }''', '''    int64_t pin_cap = (int64_t) (env_d("STRATA_XE_PINNED_GIB", 16.0) * kGiB);
    // prompt prefetch: two pinned stage buffers, each the largest layer's pageable arena bytes up to
    // STRATA_XE_PREFETCH_STAGE_GIB (1.5), taken out of the same pinned budget (TTM caps the box's pinned memory)
    if (env_i("STRATA_XE_PREFETCH", 1)) {
        std::vector<int64_t> lb((size_t) im.g.n_layers, 0);
        for (const auto& le : in_arena) lb[(size_t) le.first] += (int64_t) ((im.bbytes(le.first, le.second) + 255) & ~(size_t) 255);
        const int64_t mx = lb.empty() ? 0 : *std::max_element(lb.begin(), lb.end());
        const size_t want = (size_t) std::min<int64_t>(mx, (int64_t) (env_d("STRATA_XE_PREFETCH_STAGE_GIB", 1.5) * kGiB));
        for (auto& sl : im.pf) {
            sl.stage = want ? sycl::malloc_host<uint8_t>(want, im.ctx) : nullptr;
            if (!sl.stage) { im.pf_stage_cap = 0; break; }
            im.pf_stage_cap = want;
        }
        if (!im.pf_stage_cap) for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
        pin_cap -= 2 * (int64_t) im.pf_stage_cap;
        std::fprintf(stderr, "xe tier: prefetch staging 2 x %.2f GiB pinned (largest layer's arena share %.2f GiB)\\n",
                     im.pf_stage_cap / kGiB, mx / kGiB);
    }
    int64_t pinned = 0;
    std::vector<uint8_t> chunk_pinned(nchunks, 0);
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        if (pinned + (int64_t) chunk_size[c] <= pin_cap) {
            p = sycl::malloc_host<uint8_t>(chunk_size[c], im.ctx);
            if (p) { pinned += (int64_t) chunk_size[c]; chunk_pinned[c] = 1; }
        }'''),
    ('''    for (size_t i = 0; i < in_arena.size(); ++i)
        im.aptr[(size_t) im.key(in_arena[i].first, in_arena[i].second)] = im.achunks[where[i] >> 40] + (where[i] & (((size_t) 1 << 40) - 1));
''', '''    im.apin.assign(im.aptr.size(), 0);
    for (size_t i = 0; i < in_arena.size(); ++i) {
        const size_t key = (size_t) im.key(in_arena[i].first, in_arena[i].second);
        im.aptr[key] = im.achunks[where[i] >> 40] + (where[i] & (((size_t) 1 << 40) - 1));
        im.apin[key] = chunk_pinned[where[i] >> 40];
    }
'''),
    # ---- run_big: consume the layer's prefetch, start the next one
    ('''    // which card computes each expert; streamed experts go to the card with the fewest streamed bytes
    std::vector<int64_t> sbytes(NC, 0);''', '''    // prompt prefetch: this layer's slot (its copies enqueued now if they were not yet), and the next layer's copies
    // if they are staged already (else at the end of this call)
    Ds41XeImpl::PfSlot* ps = nullptr;
    int pslot = -1;
    if (im.pf_on) {
        for (int s2 = 0; s2 < 2; ++s2) if (im.pf[s2].layer == l) { pslot = s2; ps = &im.pf[s2]; }
        if (ps) pf_enqueue(im, pslot, true);
        if (ps && im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, false);
    }
    // which card computes each expert; streamed experts go to the card with the fewest streamed bytes (a prefetched
    // one is computed where it was prefetched to)
    std::vector<int64_t> sbytes(NC, 0);
    int64_t n_pf = 0;'''),
    ('''    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || im.card_of[(size_t) im.key(l, e)] >= 0) continue;
        size_t best = 0;''', '''    for (int64_t e = 0; e < E; ++e) {
        if (by_e[(size_t) e].empty() || im.card_of[(size_t) im.key(l, e)] >= 0) continue;
        if (ps && ps->card[(size_t) e] >= 0) { cl[(size_t) ps->card[(size_t) e]].push_back((int32_t) e); ++n_pf; continue; }
        size_t best = 0;'''),
    ('''        double tp = now_ms();
        auto lap = [&](std::atomic<int64_t>& acc) {''', '''        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<std::mutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
        }
        double tp = now_ms();
        auto lap = [&](std::atomic<int64_t>& acc) {'''),
    ('''            auto dptr_of = [&](int32_t e) -> const uint8_t* {
                const size_t key = (size_t) im.key(l, e);
                return im.card_of[key] >= 0 ? im.dptr[key] : c.stream + soff[(size_t) e];
            };
            auto in_round = [&](int32_t e) { return im.card_of[(size_t) im.key(l, e)] >= 0 ? r == 0 : round_of[(size_t) e] == r; };''',
     '''            auto dptr_of = [&](int32_t e) -> const uint8_t* {
                const size_t key = (size_t) im.key(l, e);
                if (im.card_of[key] >= 0) return im.dptr[key];
                if (ps && ps->card[(size_t) e] >= 0) return c.pbuf[pslot] + ps->off[(size_t) e];
                return c.stream + soff[(size_t) e];
            };
            auto in_round = [&](int32_t e) {
                return im.card_of[(size_t) im.key(l, e)] >= 0 || (ps && ps->card[(size_t) e] >= 0) ? r == 0 : round_of[(size_t) e] == r;
            };'''),
    ('''    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
    im.st.hits += hits;
    im.st.pcie += pcie;''', '''    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
    im.st.hits += hits;
    im.st.pcie += pcie + n_pf;
    if (ps) {   // the next layer's copies (if not on the queues yet), then this slot takes the layer after it
        im.pf_used += n_pf;
        if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, true);
        ps->layer = -1;
        if (l + 2 <= im.pf_last) pf_plan(im, pslot, l + 2);
    }'''),
    ])
    # ---- the helpers (before run_big) and the API (before close)
    patch(cp, [('''/// n tokens of one layer, expert-major.  out_host (n x n_embd) or out_dev (the dense card).''',
    '''/// Prompt prefetch: plans layer `l` into slot `s` - every arena expert of the layer (pinned ones copied as they are,
/// pageable ones staged first, while the stage has room) onto the expert card with the fewest planned bytes that has
/// room in its buffer - and asks the stager thread for the pageable copies.
void pf_plan(Ds41XeImpl& im, int s, int64_t l) {
    Ds41XeImpl::PfSlot& ps = im.pf[s];
    const int64_t E = im.g.n_experts;
    ps.layer = l;
    ps.enqueued = false;
    ps.card.assign((size_t) E, -1);
    ps.off.assign((size_t) E, 0);
    ps.soff.assign((size_t) E, SIZE_MAX);
    ps.n = 0;
    ps.bytes = 0;
    std::vector<size_t> used(im.cards.size(), 0);
    size_t sused = 0;
    for (int64_t e = 0; e < E; ++e) {
        const size_t key = (size_t) im.key(l, e);
        if (im.card_of[key] >= 0 || !im.aptr[key]) continue;
        const size_t b = (im.bbytes(l, e) + 255) & ~(size_t) 255;
        const bool pin = im.apin[key] != 0;
        if (!pin && sused + b > im.pf_stage_cap) continue;
        int best = -1;
        for (auto& cp : im.cards) {
            const XeCard& c = *cp;
            if (c.dense || !c.pbuf[s] || used[(size_t) c.idx] + b > c.pbuf_cap) continue;
            if (best < 0 || used[(size_t) c.idx] < used[(size_t) best]) best = c.idx;
        }
        if (best < 0) continue;
        ps.card[(size_t) e] = (int8_t) best;
        ps.off[(size_t) e] = used[(size_t) best];
        used[(size_t) best] += b;
        if (!pin) { ps.soff[(size_t) e] = sused; sused += b; }
        ++ps.n;
        ps.bytes += (int64_t) b;
    }
    im.pf_planned += ps.n;
    im.pf_bytes += ps.bytes;
    std::lock_guard<std::mutex> lk(im.pf_mu);
    ps.staged = sused > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
}

/// The stager thread: copies a planned slot's pageable experts into its pinned stage (no SYCL calls here: the Level
/// Zero driver must not see a second submitting thread).
void pf_stager(Ds41XeImpl& im) {
    for (;;) {
        int s = -1;
        {
            std::unique_lock<std::mutex> lk(im.pf_mu);
            im.pf_cv.wait(lk, [&] { return im.pf_stop || im.pf[0].staged == 1 || im.pf[1].staged == 1; });
            if (im.pf_stop) return;
            // the older request first (the lower layer)
            if (im.pf[0].staged == 1 && im.pf[1].staged == 1) s = im.pf[0].layer <= im.pf[1].layer ? 0 : 1;
            else s = im.pf[0].staged == 1 ? 0 : 1;
        }
        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        for (int64_t e = 0; e < im.g.n_experts; ++e)
            if (ps.card[(size_t) e] >= 0 && ps.soff[(size_t) e] != SIZE_MAX)
                std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        {
            std::lock_guard<std::mutex> lk(im.pf_mu);
            ps.staged = 0;
        }
        im.pf_cv.notify_all();
    }
}

/// Puts slot `s`'s copies on the cards' copy queues: waits for its staging when `block`, else returns if it is not
/// staged yet.
void pf_enqueue(Ds41XeImpl& im, int s, bool block) {
    Ds41XeImpl::PfSlot& ps = im.pf[s];
    if (ps.layer < 0 || ps.enqueued) return;
    {
        std::unique_lock<std::mutex> lk(im.pf_mu);
        if (ps.staged && !block) return;
        im.pf_cv.wait(lk, [&] { return ps.staged == 0; });
    }
    std::lock_guard<std::mutex> lk(g_submit);
    const int64_t l = ps.layer;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        const int c = ps.card[(size_t) e];
        if (c < 0) continue;
        XeCard& cd = *im.cards[(size_t) c];
        const uint8_t* src = ps.soff[(size_t) e] != SIZE_MAX ? ps.stage + ps.soff[(size_t) e] : im.aptr[(size_t) im.key(l, e)];
        cd.pev[s] = cd.pq.memcpy(cd.pbuf[s] + ps.off[(size_t) e], src, im.bbytes(l, e));
    }
    ps.enqueued = true;
}

/// n tokens of one layer, expert-major.  out_host (n x n_embd) or out_dev (the dense card).'''),
    ('''// ================================ the rest ================================

void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;''', '''// ================================ the rest ================================

bool Ds41XeTier::prefetch_begin(int64_t first, int64_t last) {
    Ds41XeImpl& im = *im_;
    static const int on = env_i("STRATA_XE_PREFETCH", 1);
    if (!on || !im.pf_stage_cap || im.pf_on || last < first) return false;
    // the buffers: per expert card two of the largest layer's per-card share, within its free VRAM less a reserve
    // for run_big's own buffers (STRATA_XE_PREFETCH_RESERVE_GIB, 1.3; the xe driver spills an over-commit into host
    // RAM instead of failing, so stay well inside)
    int64_t need = 0;
    int ncards = 0;
    for (auto& cp : im.cards) if (!cp->dense) ++ncards;
    if (!ncards) return false;
    for (int64_t l = first; l <= last; ++l) {
        int64_t b = 0;
        for (int64_t e = 0; e < im.g.n_experts; ++e) {
            const size_t key = (size_t) im.key(l, e);
            if (im.card_of[key] < 0 && im.aptr[key]) b += (int64_t) ((im.bbytes(l, e) + 255) & ~(size_t) 255);
        }
        need = std::max(need, b);
    }
    const double reserve = env_d("STRATA_XE_PREFETCH_RESERVE_GIB", 1.3);
    const size_t per = (size_t) (need / ncards + 2 * (int64_t) im.blob_max);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        if (c.stream) { c.q.wait(); sycl::free(c.stream, c.q); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        const int64_t fr = c.free_vram();
        int64_t room = fr > 0 ? (fr - (int64_t) (reserve * kGiB)) / 2 : 0;
        const size_t cap = (size_t) std::max<int64_t>(0, std::min<int64_t>((int64_t) per, room)) & ~(size_t) 255;
        if (cap < (size_t) im.blob_max) { c.pbuf_cap = 0; continue; }
        c.pbuf[0] = sycl::malloc_device<uint8_t>(cap, c.q);
        c.pbuf[1] = c.pbuf[0] ? sycl::malloc_device<uint8_t>(cap, c.q) : nullptr;
        if (!c.pbuf[1]) {
            if (c.pbuf[0]) sycl::free(c.pbuf[0], c.q);
            c.pbuf[0] = nullptr;
            c.pbuf_cap = 0;
            continue;
        }
        c.pbuf_cap = cap;
    }
    if (!im.pf_thr.joinable()) {
        im.pf_stop = false;
        im.pf_thr = std::thread([&im] { pf_stager(im); });
    }
    im.pf_on = true;
    im.pf_last = last;
    im.pf_used = im.pf_planned = im.pf_bytes = 0;
    im.pf[0].layer = im.pf[1].layer = -1;
    pf_plan(im, (int) (first & 1), first);
    if (first + 1 <= last) pf_plan(im, (int) ((first + 1) & 1), first + 1);
    return true;
}

void Ds41XeTier::prefetch_end() {
    Ds41XeImpl& im = *im_;
    if (!im.pf_on) return;
    {
        std::unique_lock<std::mutex> lk(im.pf_mu);   // a staging still running reads the arena into its stage
        im.pf_cv.wait(lk, [&] { return im.pf[0].staged == 0 && im.pf[1].staged == 0; });
    }
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (!c.pbuf_cap) continue;
        c.pq.wait();
        c.q.wait();
        for (auto& p : c.pbuf) { if (p) sycl::free(p, c.q); p = nullptr; }
        c.pbuf_cap = 0;
    }
    if (xe_prof())
        std::fprintf(stderr, "xe prefetch: %lld experts (%.2f GB) planned, %lld used\\n", (long long) im.pf_planned,
                     im.pf_bytes / 1e9, (long long) im.pf_used);
    im.pf[0].layer = im.pf[1].layer = -1;
    im.pf_on = false;
}

void Ds41XeTier::close() {
    if (!im_) return;
    Ds41XeImpl& im = *im_;
    prefetch_end();
    if (im.pf_thr.joinable()) {
        { std::lock_guard<std::mutex> lk(im.pf_mu); im.pf_stop = true; }
        im.pf_cv.notify_all();
        im.pf_thr.join();
    }
    for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }''')])
print('tier patched')

gp = root + 'ds41_generate.cpp'
if 'prefetch_begin' not in open(gp).read():
    patch(gp, [('''                double t0 = now_ms();
                for (int l = 0; l < n_layer; ++l) {
                    if (l < 4) vram_log("batch layer", l);''', '''                // larry: the tier streams each layer's arena experts onto the expert cards ahead of its run_chunk_dev
                // (layers whose tier call has >= STRATA_XE_PREFETCH_MIN tokens; with enc-skip the layers from CL on
                // run only the phase-B columns)
                {
                    static const int pf_min = std::getenv("STRATA_XE_PREFETCH_MIN") ? std::atoi(std::getenv("STRATA_XE_PREFETCH_MIN")) : 2048;
                    const int nb = cut > 0 ? B - cut : B;
                    const int pf_last = nb >= pf_min ? n_layer - 1 : (cut > 0 && B >= pf_min ? CL - 1 : -1);
                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(0, pf_last);
                }
                double t0 = now_ms();
                for (int l = 0; l < n_layer; ++l) {
                    if (l < 4) vram_log("batch layer", l);'''),
               ('''                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)''',
                '''                tier.prefetch_end();
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)''')])
print('generate patched')
