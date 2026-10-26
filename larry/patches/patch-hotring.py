#!/usr/bin/env python3
# patch-hotring.py [TIER_CPP] - larry, STRATA_XE_HOTRING=1 (default 0 = the old copies): a prompt's pageable arena
# experts reach the cards through small pinned rings that stay in the CPU's caches - chunks of STRATA_XE_HOTRING_KB
# (1024) KiB, STRATA_XE_HOTRING_N (4) a ring, each chunk DMA'd as soon as it is copied, so the card's read of it is
# served from the L3.  Today's copies fill a 64 MiB bounce half (on demand) or a 1.5 GiB layer stage (prefetch) before
# any DMA starts: every line is out of the caches by then - read from DRAM, written back, read again by the DMA.
# bench/dqb/stage_ring (cards 1-3, 6 GiB of pageable "experts"): the stage scheme 13.0 GB/s, a 16 MiB ring with plain
# stores 21.7 (the same ring with streaming stores 13.7, a 64 MiB one 11.6).  The prefetch's stagers then put every
# planned expert of a slot on its card themselves (pinned ones as direct chunked DMAs, in the same order) and
# pf_enqueue ends a slot with a barrier on each card's copy queue.  STRATA_XE_HOTRING_CHECK=1 (diagnostics): every
# streamed / prefetched arena expert read back from its card and compared with its source.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOTRING" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:90]))
    s = s.replace(old, new)


# ---- XeCard: the on-demand ring and the check buffer
rep("""    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
""", """    uint8_t* bounce = nullptr; size_t bounce_cap = 0;      ///< pinned staging for streamed experts in pageable memory
    uint8_t* hring = nullptr;                              ///< larry, STRATA_XE_HOTRING: the card thread's pinned ring
    std::vector<sycl::event> hring_ev;                     ///< its chunks' DMAs (a slot is reused once its copy is done)
    int hring_next = 0;
    uint8_t* hchk = nullptr;                               ///< STRATA_XE_HOTRING_CHECK: [blob_max] pinned readback
""")

# ---- Ds41XeImpl: the stagers' rings + check counters
rep("""    std::vector<std::thread> pf_thr;
""", """    std::vector<std::thread> pf_thr;
    std::vector<uint8_t*> pf_rings;    ///< larry, STRATA_XE_HOTRING: a pinned ring per stager thread
    std::atomic<int64_t> hc_n{0}, hc_bad{0};   ///< STRATA_XE_HOTRING_CHECK: experts compared / differing
""")

# ---- the hot copy + ring helpers, after stream_copy
rep("""/// larry, engine1 (STRATA_XE_PC_STREAM, default 1): an arena blob into a pinned bounce with streaming stores - no
""", """/// larry, STRATA_XE_HOTRING (default 0): pageable experts onto a card through a small pinned ring that stays in the
/// CPU's caches - each chunk DMA'd as soon as it is copied (plain stores, so its lines are still in the L3 when the
/// card reads them: one DRAM trip a byte instead of three).  bench/dqb/stage_ring: 13.0 -> 21.7 GB/s onto cards 1-3.
bool hot_on() { static const bool on = env_i("STRATA_XE_HOTRING", 0) != 0; return on; }
size_t hot_chunk() {
    static const size_t c = (size_t) std::max(64, std::min(env_i("STRATA_XE_HOTRING_KB", 1024), 16384)) << 10;
    return c;
}
int hot_slots() { static const int n = std::max(2, std::min(env_i("STRATA_XE_HOTRING_N", 4), 64)); return n; }
bool hot_check_on() { static const bool on = env_i("STRATA_XE_HOTRING_CHECK", 0) != 0; return on; }

/// a copy that leaves the destination's lines in the caches (glibc's memcpy streams copies this large past them)
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
__attribute__((target("avx2")))
#endif
void hot_copy(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;
#if defined(__x86_64__) && !defined(__SYCL_DEVICE_ONLY__)
    if (((uintptr_t) dst & 31) == 0)
        for (; i + 128 <= n; i += 128) {
            const __m256i a = _mm256_loadu_si256((const __m256i*) (src + i));
            const __m256i b = _mm256_loadu_si256((const __m256i*) (src + i + 32));
            const __m256i c = _mm256_loadu_si256((const __m256i*) (src + i + 64));
            const __m256i d = _mm256_loadu_si256((const __m256i*) (src + i + 96));
            _mm256_store_si256((__m256i*) (dst + i), a);
            _mm256_store_si256((__m256i*) (dst + i + 32), b);
            _mm256_store_si256((__m256i*) (dst + i + 64), c);
            _mm256_store_si256((__m256i*) (dst + i + 96), d);
        }
#endif
    if (i < n) std::memcpy(dst + i, src + i, n - i);
}

/// larry, engine1 (STRATA_XE_PC_STREAM, default 1): an arena blob into a pinned bounce with streaming stores - no
""")

# g_submit (PrioMutex) is defined after stream_copy and before pf_plan: the ring copy goes right after it
rep("""PrioMutex g_submit;
""", """PrioMutex g_submit;

/// STRATA_XE_HOTRING: `bytes` of host memory at `src` to `dst` on queue `q` in ring chunks: a slot whose last copy is
/// still running is waited for (outside the submit lock), then filled and its DMA enqueued at once.  `pinned`: the
/// source is DMA-able as it is (the chunks go straight, in the same order).  Returns the last copy's event.
sycl::event hot_put(sycl::queue& q, uint8_t* dst, const uint8_t* src, size_t bytes, bool pinned, uint8_t* ring,
                    std::vector<sycl::event>& ev, int& next) {
    const size_t C = hot_chunk();
    const int R = hot_slots();
    if ((int) ev.size() != R) { ev.assign((size_t) R, sycl::event()); next = 0; }
    sycl::event last;
    for (size_t off = 0; off < bytes; off += C) {
        const size_t nb = std::min(C, bytes - off);
        if (pinned) {
            std::lock_guard<PrioMutex> lk(g_submit);
            last = q.memcpy(dst + off, src + off, nb);
            continue;
        }
        const int k = next;
        next = (k + 1) % R;
        ev[(size_t) k].wait();   // a default-constructed event is complete
        uint8_t* slot = ring + (size_t) k * C;
        hot_copy(slot, src + off, nb);
        std::lock_guard<PrioMutex> lk(g_submit);
        last = ev[(size_t) k] = q.memcpy(dst + off, slot, nb);
    }
    return last;
}

/// STRATA_XE_HOTRING_CHECK: an expert's bytes on its card (queue q, after its copies) against its host source
void hot_check(Ds41XeImpl& im, XeCard& c, sycl::queue& q, const uint8_t* dev, const uint8_t* src, size_t bytes,
               int64_t l, int64_t e, const char* what) {
    if (!c.hchk) {
        std::lock_guard<PrioMutex> lk(g_submit);
        c.hchk = sycl::malloc_host<uint8_t>((size_t) im.blob_max, im.ctx);
    }
    if (!c.hchk || bytes > (size_t) im.blob_max) return;
    sycl::event ev;
    {
        std::lock_guard<PrioMutex> lk(g_submit);
        ev = q.memcpy(c.hchk, dev, bytes);
    }
    ev.wait();
    const int64_t n = ++im.hc_n;
    if (std::memcmp(c.hchk, src, bytes) != 0) {
        size_t d = 0;
        for (size_t i = 0; i < bytes; ++i) d += c.hchk[i] != src[i];
        const int64_t bad = ++im.hc_bad;
        if (bad <= 20)
            std::fprintf(stderr, "xe tier: HOTRING_CHECK %s: layer %lld expert %lld on card %d: %zu of %zu bytes differ\\n", what,
                         (long long) l, (long long) e, c.idx, d, bytes);
    }
    if (n % 20000 == 0)
        std::fprintf(stderr, "xe tier: HOTRING_CHECK: %lld experts compared, %lld differ\\n", (long long) n, (long long) im.hc_bad.load());
}
""")

# ---- pf_stager: the ring path (every planned expert of the slot, pinned ones straight)
rep("""void pf_stager(Ds41XeImpl& im) {
    const int64_t E = im.g.n_experts;
    auto has_work = [&](int s) { return im.pf[s].staged == 1 && im.pf[s].next < E; };
""", """void pf_stager(Ds41XeImpl& im, int idx) {
    const int64_t E = im.g.n_experts;
    // larry, STRATA_XE_HOTRING: this stager's ring - it puts each planned expert of a slot on its card itself
    const bool hot = hot_on() && idx < (int) im.pf_rings.size() && im.pf_rings[(size_t) idx];
    std::vector<sycl::event> hev;
    int hnext = 0;
    auto has_work = [&](int s) { return im.pf[s].staged == 1 && im.pf[s].next < E; };
""")
rep("""            while (ps.next < E && !(ps.card[(size_t) ps.next] >= 0 && ps.soff[(size_t) ps.next] != SIZE_MAX)) ++ps.next;
""", """            while (ps.next < E && !(ps.card[(size_t) ps.next] >= 0 && (hot || ps.soff[(size_t) ps.next] != SIZE_MAX))) ++ps.next;
""")
rep("""        const double ts = xe_prof() ? now_ms() : 0;
        std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
""", """        const double ts = xe_prof() ? now_ms() : 0;
        if (hot) {
            XeCard& cd = *im.cards[(size_t) ps.card[(size_t) e]];
            hot_put(cd.pq, cd.pbuf[s] + ps.off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e),
                    ps.soff[(size_t) e] == SIZE_MAX, im.pf_rings[(size_t) idx], hev, hnext);
        } else {
            std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        }
""")

# ---- pf_plan: with the ring every planned expert is the stagers' (not only the pageable ones)
rep("""    ps.staged = sused > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
""", """    ps.staged = (hot_on() && !im.pf_rings.empty() ? ps.n : (int64_t) sused) > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
""")

# ---- pf_enqueue: the stagers queued the copies already - a barrier on each card's copy queue ends the slot
rep("""    ps.t_enq = now_ms();
    const int64_t l = ps.layer;
    std::unique_lock<PrioMutex> lk(g_submit, std::defer_lock);
""", """    ps.t_enq = now_ms();
    const int64_t l = ps.layer;
    if (hot_on() && !im.pf_rings.empty()) {   // STRATA_XE_HOTRING: the stagers put every copy on the queues already
        std::lock_guard<PrioMutex> lk(g_submit);
        for (auto& cp : im.cards)
            if (cp->pbuf_cap && cp->pbuf[s]) cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
        ps.enqueued = true;
        return;
    }
    std::unique_lock<PrioMutex> lk(g_submit, std::defer_lock);
""")

# ---- prefetch_begin: the stagers' rings (allocated with the threads)
rep("""        const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));
        for (int i = 0; i < ns; ++i) im.pf_thr.emplace_back([&im] { pf_stager(im); });
""", """        const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));
        if (hot_on()) {   // STRATA_XE_HOTRING: a ring per stager (all or none: pf_enqueue counts on the stagers' copies)
            std::lock_guard<PrioMutex> lk(g_submit);
            for (int i = 0; i < ns; ++i) im.pf_rings.push_back(sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx));
            bool all = true;
            for (uint8_t* p : im.pf_rings) all = all && p;
            if (!all) {
                for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
                im.pf_rings.clear();
            }
            std::fprintf(stderr, "xe prefetch: hot rings %s %d x %d x %zu KiB\\n", all ? "on," : "FAILED (old copies),", ns, hot_slots(),
                         hot_chunk() >> 10);
        }
        for (int i = 0; i < ns; ++i) im.pf_thr.emplace_back([&im, i] { pf_stager(im, i); });
""")

# ---- close: free the rings
rep("""    for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
""", """    for (auto& sl : im.pf) if (sl.stage) { sycl::free(sl.stage, im.ctx); sl.stage = nullptr; }
    for (uint8_t* p : im.pf_rings) if (p) sycl::free(p, im.ctx);
    im.pf_rings.clear();
    if (im.hc_n > 0)
        std::fprintf(stderr, "xe tier: HOTRING_CHECK: %lld experts compared, %lld differ\\n", (long long) im.hc_n.load(),
                     (long long) im.hc_bad.load());
""")
rep("""        fr(c.gscratch); fr(c.cX2); fr(c.cX2h); fr(c.bounce); fr(c.dslots); fr(c.rstage); fr(c.xt); fr(c.ht);
""", """        fr(c.gscratch); fr(c.cX2); fr(c.cX2h); fr(c.bounce); fr(c.dslots); fr(c.rstage); fr(c.xt); fr(c.ht);
        fr(c.hring); fr(c.hchk);
""")

# ---- run_big's card_work: the prefetched experts checked, the on-demand streaming through the card's ring
rep("""        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
        }
""", """        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
        }
        if (ps && ps->enqueued && hot_check_on() && last_call)   // STRATA_XE_HOTRING_CHECK: this card's prefetched experts
            for (int64_t e = 0; e < E; ++e)
                if (ps->card[(size_t) e] == c.idx)
                    hot_check(im, c, c.q, c.pbuf[pslot] + ps->off[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e), l, e,
                              "prefetch");
""")
rep("""                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else {
                        if (hused + bytes > half) flush();
""", """                    if (pinned) {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        c.q.memcpy(c.stream + soff[(size_t) e], hb, bytes);
                    } else if (hot_on() && (c.hring || (c.hring = sycl::malloc_host<uint8_t>((size_t) hot_slots() * hot_chunk(), im.ctx)))) {
                        // STRATA_XE_HOTRING: through the card thread's ring, a DMA a chunk as it is copied
                        hot_put(c.q, c.stream + soff[(size_t) e], hb, bytes, false, c.hring, c.hring_ev, c.hring_next);
                        if (hot_check_on())
                            hot_check(im, c, c.q, c.stream + soff[(size_t) e], hb, bytes, l, e, "stream");
                    } else {
                        if (hused + bytes > half) flush();
""")
open(P, "w").write(s)
print("patched", P)
