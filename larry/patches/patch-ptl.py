#!/usr/bin/env python3
# patch-ptl.py [ENGINE_DIR] - larry, STRATA_GPU_PTL=1 (diagnostic, default off): a prompt batch's GPU timeline.  A
# one-lane kernel after each step of interest writes a flag in pinned host memory and a monitor thread stamps the time
# it sees it (host clock, ~20 us late): card 0's dense spans (attention + router, finish), its tier part and the
# combine; each expert card's call (X read, prefetch barrier, GEMM rounds, job batches, push); each card's prefetch
# DMAs per layer; host steps of the pipeline (tier launch / join, pass_sync).  ptl_dump() appends every record (ring,
# tag, layer, column, host time, done time) to logs/ptl.<pid>.tsv after each batch - bench/speed/ptlsum.py reads it.
# unitrace loses most of card 0's events in a pipelined prompt (2026-10-25: 107 of ~1300 key gathers), so this is the
# card-0 view.  Also: a batch's logits taking > 300 ms are reported (one 16K prompt's took 7.7 s, abwT60).  Idempotent.
import sys

E = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine"
T, TH, DC, DH, G = E + "/xe/ds41_xe_tier.cpp", E + "/xe/ds41_xe_tier.hpp", E + "/ds41_dense.cpp", E + "/ds41_dense.hpp", E + "/ds41_generate.cpp"
src = {p: open(p).read() for p in (T, TH, DC, DH, G)}
if "STRATA_GPU_PTL" in src[T]:
    print("already patched"); sys.exit(0)


def rep(p, old, new, n=1):
    s = src[p]
    if s.count(old) != n:
        sys.exit("%s: anchor x%d (want %d): %s" % (p, s.count(old), n, old[:100]))
    src[p] = s.replace(old, new)


# ---------------------------------------------------------------- the tier: rings, monitor, stamps, dump
rep(T, """bool Ds41XeTier::tl_on() { static const bool on = env_i("STRATA_GPU_TL", 0) != 0; return on; }
""", """bool Ds41XeTier::tl_on() { static const bool on = env_i("STRATA_GPU_TL", 0) != 0; return on; }

// larry, STRATA_GPU_PTL=1 (diagnostic, patch-ptl.py): a prompt batch's GPU timeline.  Ring 0 = the dense queue,
// 1 + k = card k's tier queue, 9 + k = card k's prefetch copy queue (in-order queues: a ring's flags flip in order),
// 17 = host-only steps.  A stamp is a one-lane kernel writing its flag; the monitor thread stamps what it sees.
bool Ds41XeTier::ptl_on() { static const bool on = env_i("STRATA_GPU_PTL", 0) != 0; return on; }

namespace {
struct PtlRing {
    static constexpr uint32_t kCap = 1u << 16;
    uint32_t* flag = nullptr;   // pinned host memory; nullptr: the host ring
    struct Rec { int tag, l, a; double th, td; };
    std::vector<Rec> rec = std::vector<Rec>(kCap);
    std::atomic<uint32_t> issued{0}, seen{0};
    uint32_t done = 0;          // dumped so far
};
constexpr int kPtlRings = 18, kPtlHost = 17;
std::array<std::atomic<PtlRing*>, kPtlRings> g_ptl{};
std::mutex g_ptl_mu;
std::thread g_ptl_mon;

void ptl_monitor() {
    for (;;) {
        bool any = false;
        for (auto& ap : g_ptl) {
            PtlRing* r = ap.load(std::memory_order_acquire);
            if (!r || !r->flag) continue;
            volatile uint32_t* f = r->flag;
            uint32_t sn = r->seen.load(std::memory_order_relaxed);
            const uint32_t is = r->issued.load(std::memory_order_acquire);
            while (sn < is && f[sn % PtlRing::kCap]) {
                r->rec[sn % PtlRing::kCap].td = now_ms();
                f[sn % PtlRing::kCap] = 0;
                ++sn;
                any = true;
            }
            r->seen.store(sn, std::memory_order_release);
        }
        if (!any) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

PtlRing* ptl_ring(int ring, const sycl::context& ctx) {
    if (ring < 0 || ring >= kPtlRings) return nullptr;
    PtlRing* r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
    if (r) return r;
    std::lock_guard<std::mutex> lk(g_ptl_mu);
    r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
    if (r) return r;
    r = new PtlRing();
    if (ring != kPtlHost) {
        r->flag = sycl::malloc_host<uint32_t>(PtlRing::kCap, ctx);
        if (!r->flag) { delete r; return nullptr; }
        for (uint32_t k = 0; k < PtlRing::kCap; ++k) r->flag[k] = 0;
    }
    g_ptl[(size_t) ring].store(r, std::memory_order_release);
    if (!g_ptl_mon.joinable()) { g_ptl_mon = std::thread(ptl_monitor); g_ptl_mon.detach(); }
    return r;
}

/// a stamp on `q` (the caller holds the submit lock like every enqueue), `th` = when the host began the step
void ptl_stamp(const sycl::context& ctx, int ring, sycl::queue& q, int tag, int l, int a, double th) {
    if (!Ds41XeTier::ptl_on()) return;
    PtlRing* r = ptl_ring(ring, ctx);
    if (!r) return;
    const uint32_t i = r->issued.load(std::memory_order_relaxed);
    if (i - r->done >= PtlRing::kCap) return;   // full until the next dump
    const uint32_t k = i % PtlRing::kCap;
    r->rec[k] = {tag, l, a, th, 0.0};
    if (r->flag) {
        volatile uint32_t* f = r->flag + k;
        q.single_task([=] { *f = 1u; });
    } else {
        r->rec[k].td = th;
    }
    r->issued.store(i + 1, std::memory_order_release);
    if (!r->flag) r->seen.store(i + 1, std::memory_order_release);
}
}  // namespace

void Ds41XeTier::ptl(int tag, int l, int a, double th) {
    Ds41XeImpl& im = *im_;
    if (!ptl_on() || !im.dense_q) return;
    ptl_stamp(im.ctx, 0, *im.dense_q, tag, l, a, th);
}

void Ds41XeTier::ptl_host(int tag, int l, int a) {
    if (!ptl_on()) return;
    PtlRing* r = ptl_ring(kPtlHost, im_->ctx);
    if (!r) return;
    const uint32_t i = r->issued.load(std::memory_order_relaxed);
    if (i - r->done >= PtlRing::kCap) return;
    const double t = now_ms();
    r->rec[i % PtlRing::kCap] = {tag, l, a, t, t};
    r->issued.store(i + 1, std::memory_order_release);
    r->seen.store(i + 1, std::memory_order_release);
}

void Ds41XeTier::ptl_dump(const char* label) {
    if (!ptl_on()) return;
    const double t0 = now_ms();
    for (auto& ap : g_ptl) {   // everything stamped has run or will in a moment (the batch's logits were read)
        PtlRing* r = ap.load(std::memory_order_acquire);
        while (r && r->seen.load(std::memory_order_acquire) < r->issued.load(std::memory_order_acquire) && now_ms() - t0 < 3000)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const char* dir = std::getenv("STRATA_GPU_PTL_DIR");
    char path[512];
    std::snprintf(path, sizeof path, "%s/ptl.%d.tsv", dir ? dir : "/opt/ai/dsv41/logs", (int) getpid());
    std::FILE* f = std::fopen(path, "a");
    if (f) std::fprintf(f, "# %s %.3f\\n", label ? label : "batch", t0);
    for (int ring = 0; ring < kPtlRings; ++ring) {
        PtlRing* r = g_ptl[(size_t) ring].load(std::memory_order_acquire);
        if (!r) continue;
        const uint32_t sn = r->seen.load(std::memory_order_acquire);
        for (uint32_t k = r->done; f && k < sn; ++k) {
            const PtlRing::Rec& e = r->rec[k % PtlRing::kCap];
            std::fprintf(f, "%d\\t%d\\t%d\\t%d\\t%.3f\\t%.3f\\n", ring, e.tag, e.l, e.a, e.th, e.td);
        }
        r->done = sn;
    }
    if (f) std::fclose(f);
}
""")
# includes the code above needs
if "#include <array>" not in src[T]:
    rep(T, "#include <atomic>\n", "#include <array>\n#include <atomic>\n")
if "#include <unistd.h>" not in src[T]:
    rep(T, "#include <atomic>\n", "#include <atomic>\n#include <unistd.h>\n")
# 10: an expert card's X read done
rep(T, """            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
""", """            c.q.memcpy(c.cX, x_dev ? x_dev : x, (size_t) n * H * 4);   // x_dev: card-to-card from the dense card
            if (!c.dense) ptl_stamp(im.ctx, 1 + c.idx, c.q, 10, (int) l, n, tq0 > 0 ? tq0 : now_ms());   // larry, STRATA_GPU_PTL
""")
# 20 / 11: the call's part starts on the card's queue (card 0: the dense queue with DENSE_INLINE); the prefetch barrier
rep(T, """        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
        }
""", """        // larry, STRATA_GPU_PTL: this card's ring (card 0's part on the dense queue: ring 0)
        const int ptl_r = im.dense_q && c.q == *im.dense_q ? 0 : 1 + c.idx;
        const double ptl_t0 = Ds41XeTier::ptl_on() ? now_ms() : 0;
        if (c.dense && Ds41XeTier::ptl_on()) { std::lock_guard<PrioMutex> lk(g_submit); ptl_stamp(im.ctx, ptl_r, c.q, 20, (int) l, n, ptl_t0); }
        if (ps && ps->enqueued) {   // the prefetched experts' copies land before any kernel of this call
            std::lock_guard<PrioMutex> lk(g_submit);
            c.q.ext_oneapi_submit_barrier({c.pev[pslot]});
            ptl_stamp(im.ctx, ptl_r, c.q, 11, (int) l, n, ptl_t0);
        }
""")
# 12: a round's GEMM experts done (both GEMM forms)
rep(T, """                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
                }
                g_bp.gemm_e += (int64_t) ge.size();
""", """                {
                    std::lock_guard<PrioMutex> lk(g_submit);
                    xe::xe_gather(c.q, Yg, c.cS, c.cW, n, K, (int) H, c.cOUT, out_init);
                    out_init = true;
                    ptl_stamp(im.ctx, ptl_r, c.q, 12, (int) l, r, now_ms());   // larry, STRATA_GPU_PTL
                }
                g_bp.gemm_e += (int64_t) ge.size();
""", 2)
# 13: a round's job batches done
rep(T, """            if (!run_jobs(jobs, jslots, true) || !run_jobs(jobs_raw, jslots_raw, false)) return false;
""", """            if (!run_jobs(jobs, jslots, true) || !run_jobs(jobs_raw, jslots_raw, false)) return false;
            if (Ds41XeTier::ptl_on() && !(jobs.empty() && jobs_raw.empty())) {   // larry, STRATA_GPU_PTL
                std::lock_guard<PrioMutex> lk(g_submit);
                ptl_stamp(im.ctx, ptl_r, c.q, 13, (int) l, r, now_ms());
            }
""")
# 14: the part pushed onto the dense card; 21: card 0's part done
rep(T, """        if (!nowait) c.q.wait();   // larry, STRATA_XE_DENSE_LITE: the combine on this queue waits for the part
""", """        if (Ds41XeTier::ptl_on()) {   // larry, STRATA_GPU_PTL
            std::lock_guard<PrioMutex> lk(g_submit);
            ptl_stamp(im.ctx, ptl_r, c.q, c.dense ? 21 : 14, (int) l, n, now_ms());
        }
        if (!nowait) c.q.wait();   // larry, STRATA_XE_DENSE_LITE: the combine on this queue waits for the part
""")
# 22 / 23: the combine (queued once every expert card's part is in; its kernel)
rep(T, """        if (cmb > 0) {   // STRATA_XE_COMBINE: every busy card's part is in its slot (its thread waited for the push)
            std::lock_guard<PrioMutex> lk(g_submit);
""", """        if (cmb > 0) {   // STRATA_XE_COMBINE: every busy card's part is in its slot (its thread waited for the push)
            std::lock_guard<PrioMutex> lk(g_submit);
            const int ptl_rc = im.dense_q && dc->q == *im.dense_q ? 0 : 1 + dc->idx;   // larry, STRATA_GPU_PTL
            const double ptl_tc = Ds41XeTier::ptl_on() ? now_ms() : 0;
            ptl_stamp(im.ctx, ptl_rc, dc->q, 22, (int) l, n, ptl_tc);
            struct PtlEnd { const sycl::context& cx; int r; sycl::queue& q; int l, n; double t;
                            ~PtlEnd() { ptl_stamp(cx, r, q, 23, l, n, t); } } ptl_end{im.ctx, ptl_rc, dc->q, (int) l, n, ptl_tc};
""")
# 30: a layer's prefetch DMAs done on each card
rep(T, """            if (cp->pbuf_cap && cp->pbuf[s]) cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
""", """            if (cp->pbuf_cap && cp->pbuf[s]) {
                cp->pev[s] = cp->pq.ext_oneapi_submit_barrier();
                ptl_stamp(im.ctx, 9 + cp->idx, cp->pq, 30, (int) l, cp->idx, ps.t_enq);   // larry, STRATA_GPU_PTL
            }
""")
rep(TH, """    void tl_tag(int id);
    void tl_flush();
""", """    void tl_tag(int id);
    void tl_flush();
    /// larry, STRATA_GPU_PTL=1 (diagnostic): a prompt batch's GPU timeline - ptl() stamps the dense queue after a step
    /// (`th`: when the host began it; the caller holds the submit lock), ptl_host() a host-only step, ptl_dump() appends
    /// the batch's records to logs/ptl.<pid>.tsv (bench/speed/ptlsum.py)
    static bool ptl_on();
    void ptl(int tag, int l, int a, double th);
    void ptl_host(int tag, int l, int a);
    void ptl_dump(const char* label);
""")

# ---------------------------------------------------------------- the dense half: spans; a slow head reported
rep(DC, """void (*ds41_dense_tl_hook)(int id) = nullptr;
""", """void (*ds41_dense_tl_hook)(int id) = nullptr;
void (*ds41_dense_ptl_hook)(int tag, int il, int col0, double t0) = nullptr;   // larry, STRATA_GPU_PTL
""")
rep(DH, """extern void (*ds41_dense_tl_hook)(int id);
""", """extern void (*ds41_dense_tl_hook)(int id);
extern void (*ds41_dense_ptl_hook)(int tag, int il, int col0, double t0);   // larry, STRATA_GPU_PTL: a prompt span queued
""")
rep(DC, """bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
""", """bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    const double tptl = ds41_dense_ptl_hook ? dprof_now() : 0;   // larry, STRATA_GPU_PTL
""")
rep(DC, """    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ring_mark(im);
""", """    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ring_mark(im);
    if (ds41_dense_ptl_hook) ds41_dense_ptl_hook(1, il, col0, tptl);   // larry, STRATA_GPU_PTL
""")
rep(DC, """    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    ring_take(im);
    const double tf0 = g_dprof_on ? dprof_now() : 0;
""", """    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tptl = ds41_dense_ptl_hook ? dprof_now() : 0;   // larry, STRATA_GPU_PTL
    ring_take(im);
    const double tf0 = g_dprof_on ? dprof_now() : 0;
""")
rep(DC, """    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false, false, sl.owns_lock());
    g_dprof_slot = -1;
    ggml_free(gc);
    ring_mark(im);
""", """    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false, false, sl.owns_lock());
    g_dprof_slot = -1;
    ggml_free(gc);
    ring_mark(im);
    if (ds41_dense_ptl_hook) ds41_dense_ptl_hook(2, il, col0, tptl);   // larry, STRATA_GPU_PTL
""")
rep(DC, """bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (!fin_flush(im)) return false;
""", """bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    const double tl0 = dprof_now();   // larry: a head taking > 300 ms is reported (one 16K prompt's took 7.7 s, abwT60)
    if (!fin_flush(im)) return false;
    const double tl1 = dprof_now();
""")
rep(DC, """    const bool ok = im.compute(gf, false);
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}
""", """    const bool ok = im.compute(gf, false);
    const double tl2 = dprof_now();
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    const double tl3 = dprof_now();
    if (tl3 - tl0 > 300) {
        size_t fr = 0, tot = 0;
        if (im.backend) ggml_backend_dev_memory(ggml_backend_get_device(im.backend), &fr, &tot);
        std::fprintf(stderr, "ds41_dense: slow logits (%d rows): flush %.0f ms, graph + compute %.0f ms, read %.0f ms; %.2f GiB free\\n",
                     nrows, tl1 - tl0, tl2 - tl1, tl3 - tl2, fr / 1073741824.0);
    }
    return ok;
}
""")

# ---------------------------------------------------------------- the pipeline: hook, host steps, the dump
rep(G, """    if (Ds41XeTier::tl_on()) ds41_dense_tl_hook = [](int id) { tl_tier->tl_tag(id); };
""", """    if (Ds41XeTier::tl_on()) ds41_dense_tl_hook = [](int id) { tl_tier->tl_tag(id); };
    if (Ds41XeTier::ptl_on()) ds41_dense_ptl_hook = [](int tag, int il, int c0, double t0) { tl_tier->ptl(tag, il, c0, t0); };   // larry, STRATA_GPU_PTL
""")
rep(G, """                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
""", """                        const double t = now_ms();
#if defined(DS41_XE)
                        tier.ptl_host(41, fl, fa);   // larry, STRATA_GPU_PTL: the join's start / end
#endif
                        const bool ok = tier.run_chunk_wait();
#if defined(DS41_XE)
                        tier.ptl_host(42, fl, fa);
#endif
                        fd_tier += now_ms() - t;
""")
rep(G, """                        const double tt = now_ms();
                        dense.pass_sync();   // the part's ids / weights / fn are final (the pass ring does not wait)
""", """                        const double tt = now_ms();
#if defined(DS41_XE)
                        tier.ptl_host(43, l, a0);   // larry, STRATA_GPU_PTL: pass_sync's start / end, the launch (40)
#endif
                        dense.pass_sync();   // the part's ids / weights / fn are final (the pass ring does not wait)
#if defined(DS41_XE)
                        tier.ptl_host(44, l, a0);
#endif
""")
rep(G, """                        fl = l; fa = a0; fb = a1;
                        ptr("T", l, a0, a1, tt);
""", """                        fl = l; fa = a0; fb = a1;
#if defined(DS41_XE)
                        tier.ptl_host(40, l, a0);
#endif
                        ptr("T", l, a0, a1, tt);
""")
rep(G, """                fd_logits += now_ms() - tl;
                btr("logits");
                c0 += B;
""", """                fd_logits += now_ms() - tl;
                btr("logits");
                if (now_ms() - tl > 300) std::fprintf(stderr, "ds41_generate: the batch's logits took %.0f ms (B %d)\\n", now_ms() - tl, B);
#if defined(DS41_XE)
                if (Ds41XeTier::ptl_on()) tier.ptl_dump("batch");   // larry, STRATA_GPU_PTL
#endif
                c0 += B;
""")
for p, s in src.items():
    open(p, "w").write(s)
print("patched", ", ".join(src))
