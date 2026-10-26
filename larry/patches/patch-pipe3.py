# patch-pipe3.py - the pipelined batch loop for any batch: layers with two parts of the same split overlap, a layer
# with one part or none (a fully phase-A batch past CL) or a moving split drains the tier call in flight first
p = '/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp'
s = open(p).read()
a = s.index('                auto split_of = [&](int l, int & lo, int & mid) {')
end_marker = '                    if (!okp) { tier.prefetch_end(); return false; }\n                }\n'
b = s.index(end_marker, a) + len(end_marker)
new = '''                // the parts of layer l's columns: [lo, mid) and [mid, B), one part [lo, B) (mid < 0), or none (lo >= B)
                auto split_of = [&](int l, int & lo, int & mid) {
                    lo = (cut > 0 && l >= CL) ? cut : 0;
                    mid = lo + (int) std::lround((double) (B - lo) / 2.0 / SB) * SB;
                    if (mid <= lo || mid >= B) mid = -1;
                };
                bool pipe = pipe_env && prdev && !mc.cpu_only && bcap_n == 0, any2 = false;
                int part_max = 0;   // the largest tier call of the batch
                for (int l = 0; l < n_layer; ++l) {
                    int lo, mid;
                    split_of(l, lo, mid);
                    if (lo >= B) continue;
                    if (mid > 0) { any2 = true; part_max = std::max(part_max, std::max(mid - lo, B - mid)); }
                    else part_max = std::max(part_max, B - lo);
                }
                pipe = pipe && any2 && part_max <= piece;
                // larry: the tier streams each layer's arena experts onto the expert cards ahead of its run_chunk_dev
                // (layers whose tier call has >= STRATA_XE_PREFETCH_MIN tokens; with enc-skip the layers from CL on
                // run only the phase-B columns)
                {
                    static const int pf_min = std::getenv("STRATA_XE_PREFETCH_MIN") ? std::atoi(std::getenv("STRATA_XE_PREFETCH_MIN")) : 2048;
                    const int nb = cut > 0 ? B - cut : B;
                    const int pf_last = nb >= pf_min ? n_layer - 1 : (cut > 0 && B >= pf_min ? CL - 1 : -1);
                    if (pf_last >= 0 && !mc.cpu_only) tier.prefetch_begin(0, pf_last, pipe ? part_max : B);
                }
                if (pipe) {
                    dense.set_submit_lock(&Ds41XeTier::submit_mutex());
                    bool inflight = false;
                    int fl = -1, fa = 0, fb = 0;   // the part whose finish waits for the tier call in flight
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
                    auto PJ = [&]() -> bool {   // join the tier call in flight, then finish its part
                        if (!inflight) return true;
                        inflight = false;
                        const double t = now_ms();
                        const bool ok = tier.run_chunk_wait();
                        fd_tier += now_ms() - t;
                        if (!ok) { ferr = "tier run refused"; return false; }
                        return PF(fl, fa, fb);
                    };
                    auto PT = [&](int l, int a0, int a1, bool last) -> bool {
                        inflight = tier.run_chunk_dev_async(l, a1 - a0, dense.batch_ids() + (size_t) a0 * top_k,
                                                            dense.batch_wts() + (size_t) a0 * top_k,
                                                            dense.batch_fn() + (size_t) a0 * n_embd, prdev + (size_t) a0 * n_embd, last);
                        if (!inflight) { ferr = "tier run refused"; return false; }
                        fl = l; fa = a0; fb = a1;
                        return true;
                    };
                    // two-part layer l:  D(l,0) [T(l-1,1) runs] | join, F(l-1,1) | T(l,0) | D(l,1) [T(l,0) runs] | join,
                    // F(l,0) | T(l,1).  A one-part layer, an empty one, a moving split and layer CL drain first.
                    bool okp = true;
                    int plo = -1, pmid = -2;
                    for (int l = 0; okp && l < n_layer; ++l) {
                        int lo, mid;
                        split_of(l, lo, mid);
                        if (mid < 0 || lo != plo || mid != pmid || (cut > 0 && l == CL)) okp = PJ();
                        if (okp && cut > 0 && l == CL) {
                            const double t = now_ms();
                            for (int s = 0; okp && s < cut; s += SB)
                                if (!dense.attn_router_span_kv(l, s, std::min(SB, cut - s))) { ferr = dense.last_error(); okp = false; }
                            fd_dense += now_ms() - t;
                        }
                        if (okp && lo < B) {
                            if (mid < 0) okp = PD(l, lo, B) && PT(l, lo, B, true);
                            else okp = PD(l, lo, mid) && PJ() && PT(l, lo, mid, false) && PD(l, mid, B) && PJ() && PT(l, mid, B, true);
                        }
                        plo = lo;
                        pmid = mid;
                    }
                    okp = okp && PJ();
                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
                    dense.set_submit_lock(nullptr);
                    if (!okp) { tier.prefetch_end(); return false; }
                }
'''
s = s[:a] + new + s[b:]
open(p, 'w').write(s)
print('ok')
