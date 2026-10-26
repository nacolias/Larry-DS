# patch-pipe.py - pipelined layer-major prompt batch (STRATA_PIPE): each layer's columns in two parts, the tier's
# part (run_chunk_dev_async on a tier thread) overlapping the dense card's other part; the dense half takes the tier's
# submit lock around its enqueues meanwhile.  Also: the prefetch buffers sized after run_big's own need for the call's
# token count, and a small overflow stream buffer while prefetching (an over-committed card faulted, 2026-10-20 14:40).
root = '/opt/ai/dsv41/xolotl/tools/v41/engine/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90])
        s = s.replace(old, new)
    open(path, 'w').write(s)

hp = root + 'xe/ds41_xe_tier.hpp'
patch(hp, [('''    bool prefetch_begin(int64_t first, int64_t last);
    void prefetch_end();
''', '''    bool prefetch_begin(int64_t first, int64_t last, int n_call);
    void prefetch_end();
    /// larry (pipelined prompt batch, STRATA_PIPE): run_chunk_dev on a tier thread - returns at once, run_chunk_wait()
    /// joins it (false = the call failed).  `last_of_layer`: the layer's last call (its prefetch slot is recycled then).
    bool run_chunk_dev_async(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                             bool last_of_layer);
    bool run_chunk_wait();
    /// The lock around every SYCL enqueue of the tier's threads; the dense half takes it too while a pipelined batch
    /// has a tier call in flight (the Level Zero driver has deadlocked / faulted under concurrent enqueues).
    static std::mutex& submit_mutex();
'''), ('#include <vector>\n', '#include <vector>\n#include <mutex>\n')] if '#include <mutex>' not in open(hp).read() else [('''    bool prefetch_begin(int64_t first, int64_t last);
    void prefetch_end();
''', '''    bool prefetch_begin(int64_t first, int64_t last, int n_call);
    void prefetch_end();
    /// larry (pipelined prompt batch, STRATA_PIPE): run_chunk_dev on a tier thread - returns at once, run_chunk_wait()
    /// joins it (false = the call failed).  `last_of_layer`: the layer's last call (its prefetch slot is recycled then).
    bool run_chunk_dev_async(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                             bool last_of_layer);
    bool run_chunk_wait();
    /// The lock around every SYCL enqueue of the tier's threads; the dense half takes it too while a pipelined batch
    /// has a tier call in flight (the Level Zero driver has deadlocked / faulted under concurrent enqueues).
    static std::mutex& submit_mutex();
''')])

cp = root + 'xe/ds41_xe_tier.cpp'
patch(cp, [
    ('''    std::condition_variable pf_cv;
    bool pf_stop = false;
''', '''    std::condition_variable pf_cv;
    bool pf_stop = false;
    // pipelined prompt batch: the tier call in flight (run_chunk_dev_async)
    std::thread run_thr;
    bool run_ok = true;
'''),
    ('''bool run_big(Ds41XeImpl& im, int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out_host,
             void* out_dev) {''', '''bool run_big(Ds41XeImpl& im, int64_t l, int n, const int32_t* ids, const float* w, const float* x, float* out_host,
             void* out_dev, bool last_call = true) {'''),
    ('''    const size_t stream_cap = (size_t) (env_d("STRATA_XE_STREAM_GIB", 1.0) * kGiB);''',
     '''    size_t stream_cap = (size_t) (env_d("STRATA_XE_STREAM_GIB", 1.0) * kGiB);
    if (im.pf_on) stream_cap = std::min(stream_cap, (size_t) (env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25) * kGiB));   // the overflow'''),
    # prep under the submit lock
    ('''    // ---- per card: buffers, X / W upload
    for (auto& cp : im.cards) {
        XeCard& c = *cp;''', '''    // ---- per card: buffers, X / W upload (under the submit lock: a pipelined batch's dense half enqueues concurrently)
    std::unique_lock<std::mutex> prep_lk(g_submit);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;'''),
    ('''        c.q.memcpy(c.cX, x, (size_t) n * H * 4);
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
        c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
    }
    const double t_prep = now_ms();''', '''        c.q.memcpy(c.cX, x, (size_t) n * H * 4);
        c.q.memcpy(c.cW, w, (size_t) n * K * 4);
        c.q.memset(c.cOUT, 0, (size_t) n * H * 4);
    }
    prep_lk.unlock();
    const double t_prep = now_ms();'''),
    ('''    if (ps) {   // the next layer's copies (if not on the queues yet), then this slot takes the layer after it
        im.pf_used += n_pf;
        if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, true);
        ps->layer = -1;
        if (l + 2 <= im.pf_last) pf_plan(im, pslot, l + 2);
    }''', '''    if (ps) {   // the layer's last call: the next layer's copies (if not on the queues yet), this slot takes l + 2
        im.pf_used += n_pf;
        if (last_call) {
            if (im.pf[pslot ^ 1].layer == l + 1) pf_enqueue(im, pslot ^ 1, true);
            ps->layer = -1;
            if (l + 2 <= im.pf_last) pf_plan(im, pslot, l + 2);
        }
    }'''),
    ('''        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {
            std::fprintf(stderr, "xe tier: dense card: combine buffer allocation failed\\n");
            return false;
        }
        for (size_t k = 0; k < others.size(); ++k) dc->q.memcpy(dc->cX2 + k * NE, others[k]->cOUT, NE * 4);
        const int np = (int) others.size();
        const float* parts = dc->cX2;
        const float* own = dc->busy ? dc->cOUT : nullptr;
        float* o = (float*) out_dev;
        dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
            float s = own ? own[i] : 0.0f;
            for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
            o[i] = s;
        }).wait();''', '''        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        sycl::event cev;
        {
            std::lock_guard<std::mutex> lk(g_submit);
            if (!others.empty() && !grow_dev(dc->cX2, dc->cX2_cap, NE * others.size(), dc->q)) {
                std::fprintf(stderr, "xe tier: dense card: combine buffer allocation failed\\n");
                return false;
            }
            for (size_t k = 0; k < others.size(); ++k) dc->q.memcpy(dc->cX2 + k * NE, others[k]->cOUT, NE * 4);
            const int np = (int) others.size();
            const float* parts = dc->cX2;
            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                float s = own ? own[i] : 0.0f;
                for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
                o[i] = s;
            });
        }
        cev.wait();'''),
    # sizing: run_big's own need for n_call tokens first
    ('''bool Ds41XeTier::prefetch_begin(int64_t first, int64_t last) {''', '''bool Ds41XeTier::prefetch_begin(int64_t first, int64_t last, int n_call) {'''),
    ('''    const double reserve = env_d("STRATA_XE_PREFETCH_RESERVE_GIB", 1.3);
    const size_t per = (size_t) (need / ncards + 2 * (int64_t) im.blob_max);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        if (c.stream) { c.q.wait(); sycl::free(c.stream, c.q); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        const int64_t fr = c.free_vram();
        int64_t room = fr > 0 ? (fr - (int64_t) (reserve * kGiB)) / 2 : 0;''', '''    // run_big's buffers for n_call tokens (an upper bound: a card's GEMM rows taken as half of all, its largest expert
    // as half the tokens), less what the card holds already; then the small overflow stream buffer and a margin
    const int64_t H = im.g.n_embd, F = im.g.n_ff, K = im.g.top_k, nc = std::max(1, n_call);
    const int64_t BJ = env_i("STRATA_XE_CHUNK_JOBS", 512), gp = nc * K / 2, gm = nc / 2 + 1;
    const int64_t rb = 2 * nc * H * 4 + 2 * nc * K * 4 + BJ * kMaxJobTok * (F + H) * 4 + nc * H * 2 + BJ * kMaxJobTok * F * 2 +
                       gm * H * 2 + 3 * F * H * 2 + gm * 2 * F * 4 + gm * F * 2 + gp * H * 4 + gp * 4;
    const double margin = env_d("STRATA_XE_PREFETCH_MARGIN_GIB", 0.5) + env_d("STRATA_XE_PREFETCH_STREAM_GIB", 0.25);
    const size_t per = (size_t) (need / ncards + 2 * (int64_t) im.blob_max);
    for (auto& cp : im.cards) {
        XeCard& c = *cp;
        if (c.dense) continue;
        if (c.stream) { c.q.wait(); sycl::free(c.stream, c.q); c.stream = nullptr; c.stream_bytes = 0; }   // regrown on demand
        const int64_t have = (int64_t) (c.cX_cap + c.cOUT_cap) * 4 + (int64_t) c.cS_cap * 8 + (int64_t) c.gscratch_cap +
                             (c.cjobs_cap ? BJ * kMaxJobTok * (F + H) * 4 : 0) + (int64_t) c.cxq_rows * H * 2;
        const int64_t reserve = std::max<int64_t>(0, rb - have) + (int64_t) (margin * kGiB);
        const int64_t fr = c.free_vram();
        int64_t room = fr > 0 ? (fr - reserve) / 2 : 0;'''),
    ('''void Ds41XeTier::prefetch_end() {''', '''std::mutex& Ds41XeTier::submit_mutex() { return g_submit; }

bool Ds41XeTier::run_chunk_dev_async(int64_t l, int n, const int32_t* ids, const float* w, const float* x, void* out_dev,
                                     bool last_of_layer) {
    Ds41XeImpl& im = *im_;
    if (im.run_thr.joinable()) return false;   // one call in flight
    im.run_ok = true;
    im.run_thr = std::thread([&im, l, n, ids, w, x, out_dev, last_of_layer] {
        try {
            im.run_ok = n <= kMaxJobTok ? run_small(im, l, n, ids, w, x, nullptr, out_dev)
                                        : run_big(im, l, n, ids, w, x, nullptr, out_dev, last_of_layer);
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "xe tier: run_chunk_dev_async (layer %lld, %d tokens): %s\\n", (long long) l, n, e.what());
            im.run_ok = false;
        }
    });
    return true;
}

bool Ds41XeTier::run_chunk_wait() {
    Ds41XeImpl& im = *im_;
    if (!im.run_thr.joinable()) return true;
    im.run_thr.join();
    return im.run_ok;
}

void Ds41XeTier::prefetch_end() {'''),
    ('''    prefetch_end();
    if (im.pf_thr.joinable()) {''', '''    if (im.run_thr.joinable()) im.run_thr.join();
    prefetch_end();
    if (im.pf_thr.joinable()) {'''),
])
print('tier patched')

dh = root + 'ds41_dense.hpp'
s = open(dh).read()
if 'set_submit_lock' not in s:
    s = s.replace('#include <memory>\n', '#include <memory>\n#include <mutex>\n', 1)
    old = '''    bool finish_span(int il, int col0, int n);
'''
    assert s.count(old) == 1
    s = s.replace(old, old + '''    /// larry (pipelined prompt batch): while set, the span calls (attn_router_span[_kv], finish_span) hold this lock
    /// around their enqueues (not their waits) - the tier's card threads enqueue concurrently
    void set_submit_lock(std::mutex * m);
''', 1)
    open(dh, 'w').write(s)

dc = root + 'ds41_dense.cpp'
patch(dc, [
    ('''    bool compute(ggml_cgraph * gf, bool cached, bool async = false);''',
     '''    bool compute(ggml_cgraph * gf, bool cached, bool async = false, bool nosync = false);
    std::mutex * sub_mu = nullptr;            // larry: set_submit_lock'''),
    ('''bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached, bool async) {''',
     '''bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached, bool async, bool nosync) {'''),
    ('''    const ggml_status st = async ? ggml_backend_graph_compute_async(backend, gf) : ggml_backend_graph_compute(backend, gf);''',
     '''    const ggml_status st = async || nosync ? ggml_backend_graph_compute_async(backend, gf) : ggml_backend_graph_compute(backend, gf);'''),
    # attn_router_span_kv
    ('''bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;''', '''bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);'''),
    ('''    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;''', '''    const bool okk = im.compute(gk, false, false, sl.owns_lock());
    ggml_free(gck);
    if (sl.owns_lock()) { sl.unlock(); ggml_backend_synchronize(im.backend); }
    return okk;'''),
    # attn_router_span
    ('''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;''', '''bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);'''),
    ('''    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_synchronize(im.backend);
    lap(4);
    return true;
}''', '''    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    if (sl.owns_lock()) sl.unlock();
    ggml_backend_synchronize(im.backend);
    lap(4);
    return true;
}

void Ds41Dense::set_submit_lock(std::mutex * m) { p_->sub_mu = m; }'''),
    # finish_span
    ('''    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;''', '''    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    std::unique_lock<std::mutex> sl;
    if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);
    const double tf0 = g_dprof_on ? dprof_now() : 0;'''),
    ('''    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;''', '''    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false, false, sl.owns_lock());
    g_dprof_slot = -1;
    ggml_free(gc);
    if (sl.owns_lock()) { sl.unlock(); ggml_backend_synchronize(im.backend); }
    return ok;'''),
])
# attn_router_span's own compute: nosync while locked (the readbacks follow in queue order, then the wait)
s = open(dc).read()
i = s.index('bool Ds41Dense::attn_router_span(int il, int col0, int n) {')
j = s.index('    const bool ok = im.compute(gf, false);', i)
s = s[:j] + '    const bool ok = im.compute(gf, false, false, sl.owns_lock());' + s[j + len('    const bool ok = im.compute(gf, false);'):]
open(dc, 'w').write(s)
print('dense patched')

gp = root + 'ds41_generate.cpp'
patch(gp, [
    ('''                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(0, pf_last);
                }''', '''                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(0, pf_last, std::max(B / 2, nb / 2) + SB);
                }
                // larry, STRATA_PIPE (default 1): every layer's columns in two parts, the tier's part overlapping the
                // dense card's other part:  D(l,0) | join T(l-1,1), F(l-1,1) | launch T(l,0) | D(l,1) | join T(l,0),
                // F(l,0) | launch T(l,1).  With enc-skip the split moves at layer CL (the previous layer finishes
                // completely first).  The tier's card threads then enqueue while the dense half does, so the dense
                // half takes the tier's submit lock around its enqueues.
                static const bool pipe_env = !(std::getenv("STRATA_PIPE") && std::getenv("STRATA_PIPE")[0] == '0');
                float * prdev = routed_host ? nullptr : dense.routed_dev();
                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                };
                bool pipe = pipe_env && prdev && piece >= B && !mc.cpu_only && bcap_n == 0;
                for (int l = 0; pipe && l < n_layer; ++l) { int lo, mid; split_of(l, lo, mid); if (mid < 0) pipe = false; }
                if (pipe) {
                    dense.set_submit_lock(&Ds41XeTier::submit_mutex());
                    bool inflight = false;
                    auto PD = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.attn_router_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        return true;
                    };
                    auto PF = [&](int l, int a0, int a1) -> bool {
                        const double t = now_ms();
                        for (int s = a0; s < a1; s += SB)
                            if (!dense.finish_span(l, s, std::min(SB, a1 - s))) { ferr = dense.last_error(); return false; }
                        fd_dense += now_ms() - t;
                        return true;
                    };
                    auto PJ = [&]() -> bool {
                        if (!inflight) return true;
                        inflight = false;
                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        if (!ok) ferr = "tier run refused";
                        return ok;
                    };
                    auto PT = [&](int l, int a0, int a1, bool last) -> bool {
                        inflight = tier.run_chunk_dev_async(l, a1 - a0, dense.batch_ids() + (size_t) a0 * top_k,
                                                            dense.batch_wts() + (size_t) a0 * top_k,
                                                            dense.batch_fn() + (size_t) a0 * n_embd, prdev + (size_t) a0 * n_embd, last);
                        if (!inflight) ferr = "tier run refused";
                        return inflight;
                    };
                    bool okp = true;
                    int plo = 0, pmid = 0;
                    for (int l = 0; okp && l < n_layer; ++l) {
                        int lo, mid;
                        split_of(l, lo, mid);
                        if (l > 0 && (lo != plo || mid != pmid)) okp = PJ() && PF(l - 1, pmid, B);   // the split moves
                        if (okp && cut > 0 && l == CL) {
                            const double t = now_ms();
                            for (int s = 0; okp && s < cut; s += SB)
                                if (!dense.attn_router_span_kv(l, s, std::min(SB, cut - s))) { ferr = dense.last_error(); okp = false; }
                            fd_dense += now_ms() - t;
                        }
                        okp = okp && PD(l, lo, mid);
                        if (okp && inflight) okp = PJ() && PF(l - 1, pmid, B);
                        okp = okp && PT(l, lo, mid, false) && PD(l, mid, B) && PJ() && PF(l, lo, mid) && PT(l, mid, B, true);
                        plo = lo;
                        pmid = mid;
                    }
                    okp = okp && PJ() && PF(n_layer - 1, pmid, B);
                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
                    dense.set_submit_lock(nullptr);
                    if (!okp) { tier.prefetch_end(); return false; }
                }'''),
    ('''                double t0 = now_ms();
                for (int l = 0; l < n_layer; ++l) {
                    if (l < 4) vram_log("batch layer", l);''', '''                double t0 = now_ms();
                for (int l = 0; !pipe && l < n_layer; ++l) {
                    if (l < 4) vram_log("batch layer", l);'''),
])
print('generate patched')
