# patch-pf2.py - prefetch staging on STRATA_XE_STAGERS (4) threads; prefetch copies enqueued 16 per submit-lock hold;
# the prefetch buffers logged; dense prof: submit-lock waits + finish time, printed after a one-shot prompt too
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

cp = root + 'xe/ds41_xe_tier.cpp'
patch(cp, [
    ('''        int staged = 0;                ///< 0 nothing to stage / done, 1 requested (stager thread), under pf_mu''',
     '''        int staged = 0;                ///< 0 nothing to stage / done, 1 requested (stager threads), under pf_mu
        int64_t next = 0;              ///< the next expert a stager takes (under pf_mu)
        int active = 0;                ///< stagers copying for it (under pf_mu)'''),
    ('''    std::thread pf_thr;
    std::mutex pf_mu;''', '''    std::vector<std::thread> pf_thr;
    std::mutex pf_mu;'''),
    ('''    std::lock_guard<std::mutex> lk(im.pf_mu);
    ps.staged = sused > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
}''', '''    std::lock_guard<std::mutex> lk(im.pf_mu);
    ps.next = 0;
    ps.active = 0;
    ps.staged = sused > 0 ? 1 : 0;
    if (ps.staged) im.pf_cv.notify_all();
}'''),
    ('''/// The stager thread: copies a planned slot's pageable experts into its pinned stage (no SYCL calls here: the Level
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
}''', '''/// A stager thread (STRATA_XE_STAGERS of them): copies planned slots' pageable experts into their pinned stage, an
/// expert at a time, the older request (lower layer) first; whoever finishes a slot's last expert marks it staged.
/// No SYCL calls here: the Level Zero driver must not see another submitting thread.
void pf_stager(Ds41XeImpl& im) {
    const int64_t E = im.g.n_experts;
    auto has_work = [&](int s) { return im.pf[s].staged == 1 && im.pf[s].next < E; };
    for (;;) {
        int s = -1;
        int64_t e = -1;
        {
            std::unique_lock<std::mutex> lk(im.pf_mu);
            im.pf_cv.wait(lk, [&] { return im.pf_stop || has_work(0) || has_work(1); });
            if (im.pf_stop) return;
            if (has_work(0) && has_work(1)) s = im.pf[0].layer <= im.pf[1].layer ? 0 : 1;
            else s = has_work(0) ? 0 : 1;
            Ds41XeImpl::PfSlot& ps = im.pf[s];
            while (ps.next < E && !(ps.card[(size_t) ps.next] >= 0 && ps.soff[(size_t) ps.next] != SIZE_MAX)) ++ps.next;
            if (ps.next < E) { e = ps.next++; ++ps.active; }
            if (e < 0 && ps.active == 0) { ps.staged = 0; im.pf_cv.notify_all(); continue; }
            if (e < 0) continue;
        }
        Ds41XeImpl::PfSlot& ps = im.pf[s];
        const int64_t l = ps.layer;
        std::memcpy(ps.stage + ps.soff[(size_t) e], im.aptr[(size_t) im.key(l, e)], im.bbytes(l, e));
        std::lock_guard<std::mutex> lk(im.pf_mu);
        --ps.active;
        if (ps.active == 0 && ps.next >= E) { ps.staged = 0; im.pf_cv.notify_all(); }
    }
}'''),
    ('''    std::lock_guard<std::mutex> lk(g_submit);
    const int64_t l = ps.layer;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        const int c = ps.card[(size_t) e];
        if (c < 0) continue;
        XeCard& cd = *im.cards[(size_t) c];
        const uint8_t* src = ps.soff[(size_t) e] != SIZE_MAX ? ps.stage + ps.soff[(size_t) e] : im.aptr[(size_t) im.key(l, e)];
        cd.pev[s] = cd.pq.memcpy(cd.pbuf[s] + ps.off[(size_t) e], src, im.bbytes(l, e));
    }
    ps.enqueued = true;''', '''    const int64_t l = ps.layer;
    std::unique_lock<std::mutex> lk(g_submit, std::defer_lock);
    int held = 0;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        const int c = ps.card[(size_t) e];
        if (c < 0) continue;
        if (!lk.owns_lock()) lk.lock();
        XeCard& cd = *im.cards[(size_t) c];
        const uint8_t* src = ps.soff[(size_t) e] != SIZE_MAX ? ps.stage + ps.soff[(size_t) e] : im.aptr[(size_t) im.key(l, e)];
        cd.pev[s] = cd.pq.memcpy(cd.pbuf[s] + ps.off[(size_t) e], src, im.bbytes(l, e));
        if (++held == 16) { lk.unlock(); held = 0; }   // the other threads' enqueues get in between
    }
    ps.enqueued = true;'''),
    ('''    if (!im.pf_thr.joinable()) {
        im.pf_stop = false;
        im.pf_thr = std::thread([&im] { pf_stager(im); });
    }''', '''    if (im.pf_thr.empty()) {
        im.pf_stop = false;
        const int ns = std::max(1, env_i("STRATA_XE_STAGERS", 4));
        for (int i = 0; i < ns; ++i) im.pf_thr.emplace_back([&im] { pf_stager(im); });
    }
    static bool logged = false;
    if (!logged || xe_prof()) {
        logged = true;
        std::fprintf(stderr, "xe prefetch: layers %lld..%lld, %d tokens a call, largest layer %.2f GiB; buffers per card 2 x", (long long) first,
                     (long long) last, n_call, need / kGiB);
        for (auto& cp : im.cards) if (!cp->dense) std::fprintf(stderr, " %.2f", cp->pbuf_cap / kGiB);
        std::fprintf(stderr, " GiB\\n");
    }'''),
    ('''    if (im.pf_thr.joinable()) {
        { std::lock_guard<std::mutex> lk(im.pf_mu); im.pf_stop = true; }
        im.pf_cv.notify_all();
        im.pf_thr.join();
    }''', '''    if (!im.pf_thr.empty()) {
        { std::lock_guard<std::mutex> lk(im.pf_mu); im.pf_stop = true; }
        im.pf_cv.notify_all();
        for (auto& t : im.pf_thr) t.join();
        im.pf_thr.clear();
    }'''),
])
print('tier patched')

dc = root + 'ds41_dense.cpp'
patch(dc, [
    ('''static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls''',
     '''static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double g_dlock = 0, g_dfin = 0;   // larry: submit-lock waits (pipelined batch), finish_span wall'''),
    ('''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);''', '''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) {
        const double tl = g_dprof_on ? dprof_now() : 0;
        sl = std::unique_lock<std::mutex>(*im.sub_mu);
        if (g_dprof_on) g_dlock += dprof_now() - tl;
    }'''),
    ('''    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);
    const double tf0 = g_dprof_on ? dprof_now() : 0;''', '''    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) {
        sl = std::unique_lock<std::mutex>(*im.sub_mu);
        if (g_dprof_on) g_dlock += dprof_now() - tf0;
    }'''),
    ('''    if (sl.owns_lock()) { sl.unlock(); ggml_backend_synchronize(im.backend); }
    return ok;''', '''    if (sl.owns_lock()) { sl.unlock(); ggml_backend_synchronize(im.backend); }
    if (g_dprof_on) g_dfin += dprof_now() - tf0;
    return ok;'''),
    ('''                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;''', '''                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    std::fprintf(stderr, "dense prof %s: finish spans %.0f ms, submit-lock waits %.0f ms\\n", tag, g_dfin, g_dlock);
    for (double & v : g_dprof) v = 0;
    g_dfin = g_dlock = 0;'''),
])
print('dense patched')
gp = root + 'ds41_generate.cpp'
s = open(gp).read()
old = '''    std::fprintf(stderr, "prompt split: dense %.1f s, experts %.1f s (file wait %.1f s), logits %.1f s\\n", fd_dense / 1000.0,'''
assert s.count(old) == 1
s = s.replace(old, '''    dense.prof_report("prompt");
''' + old, 1)
open(gp, 'w').write(s)
print('generate patched')
