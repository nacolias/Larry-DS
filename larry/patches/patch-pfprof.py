# patch-pfprof.py - STRATA_XE_PROF: the prefetch's own timing - stager memcpy time / bytes, pf_enqueue's waits for
# staging, and per layer how long the cards' copies take from enqueue to done (waited for at run_big's start)
p = '/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp'
s = open(p).read()
pairs = [
    ('''    std::atomic<int64_t> lk_gemm_us{0}, lk_jobs_us{0}, lk_n{0};   ///< submit-lock hold time of the GEMM / job sites''',
     '''    std::atomic<int64_t> lk_gemm_us{0}, lk_jobs_us{0}, lk_n{0};   ///< submit-lock hold time of the GEMM / job sites
    std::atomic<int64_t> st_us{0}, st_bytes{0}, pfw_us{0}, dma_us{0}, dma_wait_us{0}, dma_bytes{0}, dma_n{0};   ///< prefetch'''),
    ('''        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));''',
     '''        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        const double ts = xe_prof() ? now_ms() : 0;
        std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        if (xe_prof()) { g_bp.st_us += (int64_t) ((now_ms() - ts) * 1000.0); g_bp.st_bytes += (int64_t) im.bbytes(l, e); }'''),
    ('''    {
        std::unique_lock<std::mutex> lk(im.pf_mu);
        if (ps.staged && !block) return;
        im.pf_cv.wait(lk, [&] { return ps.staged == 0; });
    }''', '''    {
        const double tw = xe_prof() ? now_ms() : 0;
        std::unique_lock<std::mutex> lk(im.pf_mu);
        if (ps.staged && !block) return;
        im.pf_cv.wait(lk, [&] { return ps.staged == 0; });
        if (xe_prof()) g_bp.pfw_us += (int64_t) ((now_ms() - tw) * 1000.0);
    }
    ps.t_enq = now_ms();'''),
    ('''        int64_t n = 0, bytes = 0;      ///< experts / bytes planned''',
     '''        int64_t n = 0, bytes = 0;      ///< experts / bytes planned
        double t_enq = 0;              ///< when its copies went on the queues (STRATA_XE_PROF)
        bool timed = false;'''),
    ('''        if (ps) pf_enqueue(im, pslot, true);''', '''        if (ps) pf_enqueue(im, pslot, true);
        if (ps && prof && !ps->timed) {   // the layer's copies: done how long after their enqueue (this waits for them)
            ps->timed = true;
            const double tw = now_ms();
            for (auto& cp : im.cards) if (!cp->dense && cp->pbuf_cap) cp->pev[pslot].wait();
            const double td = now_ms();
            g_bp.dma_wait_us += (int64_t) ((td - tw) * 1000.0);
            g_bp.dma_us += (int64_t) ((td - ps->t_enq) * 1000.0);
            g_bp.dma_bytes += ps->bytes;
            ++g_bp.dma_n;
        }'''),
    ('''        if (prof && g_bp.calls % 40 == 0)
            std::fprintf(stderr, "xe prof: submit-lock holds: GEMM experts %.0f ms (%lld), job batches %.0f ms\\n",
                         g_bp.lk_gemm_us / 1000.0, (long long) g_bp.lk_n.load(), g_bp.lk_jobs_us / 1000.0);''',
     '''        if (prof && g_bp.calls % 40 == 0) {
            std::fprintf(stderr, "xe prof: submit-lock holds: GEMM experts %.0f ms (%lld), job batches %.0f ms\\n",
                         g_bp.lk_gemm_us / 1000.0, (long long) g_bp.lk_n.load(), g_bp.lk_jobs_us / 1000.0);
            std::fprintf(stderr, "xe prof: prefetch: staging %.1f GB in %.0f thread-ms (%.1f GB/s a thread), enqueue waits %.0f ms; "
                         "%lld layers' copies %.1f GB, enqueue -> done %.0f ms total (%.1f GB/s), run_big waited %.0f ms\\n",
                         g_bp.st_bytes / 1e9, g_bp.st_us / 1000.0, g_bp.st_us ? g_bp.st_bytes / 1e3 / g_bp.st_us : 0.0,
                         g_bp.pfw_us / 1000.0, (long long) g_bp.dma_n.load(), g_bp.dma_bytes / 1e9, g_bp.dma_us / 1000.0,
                         g_bp.dma_us ? g_bp.dma_bytes / 1e3 / g_bp.dma_us : 0.0, g_bp.dma_wait_us / 1000.0);
        }'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
# pf_plan resets the timing flag
s = s.replace('''    ps.layer = l;
    ps.enqueued = false;''', '''    ps.layer = l;
    ps.enqueued = false;
    ps.timed = false;''', 1)
open(p, 'w').write(s)
print('ok')
