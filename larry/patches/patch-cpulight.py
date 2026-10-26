#!/usr/bin/env python3
# patch-cpulight.py [TIER_CPP] - larry, STRATA_XE_CPU_LIGHT=T (default 0 = off, T <= 8): in a prompt call (run_big), an
# arena expert the call would stream (neither prefetched nor kept in a stream buffer) that serves <= T tokens runs on
# the CPU pool beside the cards instead.  A short prompt (no prefetch below 2048 tokens) streams ~2,000 experts on
# the card threads: a pageable one is read and written by the bounce copy and read again by the DMA - the expert
# phase is bound by host memory; the pool reads it once, while the cards run the rest.  The CPU part's weighted sum
# (pinned) joins the dense card's combine as one more input.  Idempotent.  (The 2026-10-21 try - patch-light.py,
# reverted - ran beside a misplaced pool at 21-26 GB/s and in long prompts only.)
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_CPU_LIGHT" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


# ---- state: the light experts' buffers
rep("""    std::vector<float> cpart;              ///< [job][8][n_embd] CPU expert outputs
""", """    std::vector<float> cpart;              ///< [job][8][n_embd] CPU expert outputs
    // STRATA_XE_CPU_LIGHT (prompt calls, run_big): the light arena experts' CPU-pool buffers
    std::vector<cpu::ActQ> lt_actq;        ///< per token of the call
    std::vector<uint8_t> lt_nact;          ///< [token][format slot][kNativeActBytes]
    std::vector<uint8_t> lt_have;          ///< [token][1 + format slot]: quantised already this call
    std::vector<float> lt_cpart;           ///< [job][8][n_embd] their outputs
    std::vector<cpu::ExpertJobMulti> lt_jobs;
    float* lt_hs = nullptr;                ///< pinned: their weighted sum [n][n_embd]
    size_t lt_hs_cap = 0;
    float* lt_dev = nullptr;               ///< the dense card: lt_hs copied there for the combine
    size_t lt_dev_cap = 0;
    float* lt_x = nullptr;                 ///< pinned: a pipelined call's X read back off the dense card
    size_t lt_x_cap = 0;
    int64_t lt_calls = 0, lt_experts = 0, lt_streamed = 0;
    double lt_bytes = 0, lt_q_ms = 0, lt_run_ms = 0, lt_sum_ms = 0, lt_wait_ms = 0;
""")

# ---- run_big: which experts go to the CPU
rep("""    const int gemm_min = env_i("STRATA_XE_GEMM_MIN", 8);
    const bool prof = xe_prof();
    // per expert: its (token, slot) entries in routing order
""", """    const int gemm_min = env_i("STRATA_XE_GEMM_MIN", 8);
    const bool prof = xe_prof();
    // larry, STRATA_XE_CPU_LIGHT=T (0 = off, at most 8): an arena expert this call would stream (neither prefetched
    // nor kept in a stream buffer) that serves <= T tokens runs on the CPU pool beside the cards instead.  A short
    // prompt's expert phase is bound by host memory: a pageable expert streamed is read and written by the bounce copy
    // and read again by the DMA; the pool reads it once.  The pool needs the call's X on the host: a pipelined batch
    // (> 512 tokens) passes x_dev and leaves the host copy stale - read back once a call (n x 20 KB on the dense card's
    // prefetch queue, idle there).  Prompt calls only run here: the pool is idle.  STRATA_XE_CPU_LIGHT_PINNED=0:
    // pinned arena experts (the DMA's one read) stream as before.  STRATA_XE_CPU_LIGHT_MAX_N: calls of more tokens
    // stream everything (default 4096).
    static const int light_t = std::max(0, std::min(env_i("STRATA_XE_CPU_LIGHT", 0), kMaxJobTok));
    static const int light_pin = env_i("STRATA_XE_CPU_LIGHT_PINNED", 1);
    static const int light_max_n = env_i("STRATA_XE_CPU_LIGHT_MAX_N", 4096);
    const bool light_on = light_t > 0 && n <= light_max_n && (x_dev || x) && im.pool && (!x_dev || im.dense >= 0);
    std::vector<int32_t> light;   // this call's CPU-pool experts
    // per expert: its (token, slot) entries in routing order
""")
rep("""        if (keep_on && kept_card[(size_t) e] >= 0) {
            const int kc = kept_card[(size_t) e];
            cl[(size_t) kc].push_back((int32_t) e);
            kept[(size_t) kc].push_back((int32_t) e);
            continue;
        }
        size_t best = 0;
""", """        if (keep_on && kept_card[(size_t) e] >= 0) {
            const int kc = kept_card[(size_t) e];
            cl[(size_t) kc].push_back((int32_t) e);
            kept[(size_t) kc].push_back((int32_t) e);
            continue;
        }
        if (light_on && (int) by_e[(size_t) e].size() <= light_t && im.aptr[(size_t) im.key(l, e)] &&
            (light_pin || !im.apin[(size_t) im.key(l, e)])) {
            light.push_back((int32_t) e);
            continue;
        }
        size_t best = 0;
""")

# ---- the CPU part, between the card threads' start and their join
rep("""    {
        std::vector<std::thread> th;
        for (auto& cp : im.cards) {
            XeCard* c = cp.get();
            if (!c->busy) continue;
            th.emplace_back([&, c]() {
                try {
                    ok[(size_t) c->idx] = card_work(*c) ? 1 : 0;
                } catch (const sycl::exception& e) {
                    std::fprintf(stderr, "xe tier: card %d chunk: %s\\n", c->idx, e.what());
                    ok[(size_t) c->idx] = 0;
                }
            });
        }
        for (auto& t : th) t.join();
    }
    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
""", """    const size_t NE = (size_t) n * H;
    bool light_ok = true;
    {
        std::vector<std::thread> th;
        for (auto& cp : im.cards) {
            XeCard* c = cp.get();
            if (!c->busy) continue;
            th.emplace_back([&, c]() {
                try {
                    ok[(size_t) c->idx] = card_work(*c) ? 1 : 0;
                } catch (const sycl::exception& e) {
                    std::fprintf(stderr, "xe tier: card %d chunk: %s\\n", c->idx, e.what());
                    ok[(size_t) c->idx] = 0;
                }
            });
        }
        // STRATA_XE_CPU_LIGHT: the light experts on the pool while the cards run theirs (their weighted sum -> lt_hs).
        // Nothing here may throw past the card threads' join (a joinable std::thread's destructor terminates).
        if (!light.empty()) try {
            const double tq = now_ms();
            const size_t nf = im.nact_fmt.size();
            if (im.lt_actq.size() < (size_t) n) im.lt_actq.resize((size_t) n);
            if (im.lt_nact.size() < (size_t) n * std::max<size_t>(1, nf) * cpu::kNativeActBytes)
                im.lt_nact.resize((size_t) n * std::max<size_t>(1, nf) * cpu::kNativeActBytes);
            im.lt_have.assign((size_t) n * (1 + nf), 0);
            if (im.lt_jobs.size() < (size_t) cpu::ExpertPool::kMaxSplitMulti) im.lt_jobs.resize(cpu::ExpertPool::kMaxSplitMulti);
            if (im.lt_cpart.size() < light.size() * kMaxJobTok * (size_t) H) im.lt_cpart.resize(light.size() * kMaxJobTok * (size_t) H);
            if (im.lt_hs_cap < NE) {
                if (im.lt_hs) sycl::free(im.lt_hs, im.ctx);
                im.lt_hs = sycl::malloc_host<float>(NE, im.ctx);
                im.lt_hs_cap = im.lt_hs ? NE : 0;
            }
            if (!im.lt_hs) { std::fprintf(stderr, "xe tier: CPU-light sum buffer allocation failed\\n"); light_ok = false; }
            const float* xl = x;
            if (light_ok && x_dev) {   // a pipelined call: X off the dense card (its host copy is stale)
                if (im.lt_x_cap < NE) {
                    if (im.lt_x) sycl::free(im.lt_x, im.ctx);
                    im.lt_x = sycl::malloc_host<float>(NE, im.ctx);
                    im.lt_x_cap = im.lt_x ? NE : 0;
                }
                if (!im.lt_x) { std::fprintf(stderr, "xe tier: CPU-light X buffer allocation failed\\n"); light_ok = false; }
                else {
                    sycl::event xev;
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        xev = im.cards[(size_t) im.dense]->pq.memcpy(im.lt_x, x_dev, NE * 4);
                    }
                    xev.wait();
                    xl = im.lt_x;
                }
            }
            double t_run = 0, t_sum = 0;
            size_t done = 0;
            while (light_ok && done < light.size()) {
                const size_t nb = std::min<size_t>(light.size() - done, (size_t) cpu::ExpertPool::kMaxSplitMulti);
                for (size_t i = 0; i < nb; ++i) {
                    const int32_t e = light[done + i];
                    const cpu::NativeFmt& f = im.fl[(size_t) im.cls(l, e)];
                    int fs = -1;
                    if (!cpu::q2_native_kernels(f.gu_type))
                        for (size_t b = 0; b < nf; ++b) if (im.nact_fmt[b] == f.gu_act) fs = (int) b;
                    cpu::ExpertJobMulti& j = im.lt_jobs[i];
                    j = cpu::ExpertJobMulti();
                    j.blob = im.aptr[(size_t) im.key(l, e)];
                    j.fmt = &f;
                    const std::vector<int32_t>& sl = by_e[(size_t) e];
                    j.nt = (int) sl.size();
                    for (int si = 0; si < j.nt; ++si) {
                        const int t = sl[(size_t) si] / K;
                        uint8_t& hv = im.lt_have[(size_t) t * (1 + nf) + (size_t) (fs + 1)];
                        uint8_t* na = fs < 0 ? nullptr : im.lt_nact.data() + ((size_t) t * nf + (size_t) fs) * cpu::kNativeActBytes;
                        if (!hv) {
                            if (fs < 0) cpu::act_quant_any(xl + (size_t) t * H, (int) H, im.lt_actq[(size_t) t]);
                            else cpu::native_quant_act(f, xl + (size_t) t * H, na);
                            hv = 1;
                        }
                        j.act[si] = &im.lt_actq[(size_t) t];
                        j.nact[si] = na;
                        j.out[si] = im.lt_cpart.data() + ((done + i) * kMaxJobTok + (size_t) si) * H;
                    }
                    im.lt_bytes += (double) im.bbytes(l, e);
                }
                const double tr = now_ms();
                im.pool->run_split_multi_native(*im.lt_jobs[0].fmt, im.lt_jobs.data(), (int) nb);
                t_run += now_ms() - tr;
                done += nb;
            }
            if (light_ok) {   // weighted into the call's rows (routing order within a token: slot k ascending per expert)
                const double ts = now_ms();
                std::memset(im.lt_hs, 0, NE * sizeof(float));
                for (size_t i = 0; i < light.size(); ++i) {
                    const std::vector<int32_t>& sl = by_e[(size_t) light[i]];
                    for (size_t si = 0; si < sl.size(); ++si) {
                        const int t = sl[si] / K;
                        const float wt = w[(size_t) sl[si]];
                        float* o = im.lt_hs + (size_t) t * H;
                        const float* p = im.lt_cpart.data() + (i * kMaxJobTok + si) * H;
                        for (int64_t h = 0; h < H; ++h) o[h] += wt * p[h];
                    }
                }
                t_sum = now_ms() - ts;
            }
            ++im.lt_calls;
            im.lt_experts += (int64_t) light.size();
            im.lt_q_ms += now_ms() - tq - t_run - t_sum;
            im.lt_run_ms += t_run;
            im.lt_sum_ms += t_sum;
            im.st.cpu += (int64_t) light.size();
            im.st.cpu_ms += now_ms() - tq;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xe tier: CPU-light (layer %lld, %d tokens): %s\\n", (long long) l, n, e.what());
            light_ok = false;
        }
        const double tw = now_ms();
        for (auto& t : th) t.join();
        if (!light.empty()) im.lt_wait_ms += now_ms() - tw;
        for (size_t k = 0; k < NC; ++k) im.lt_streamed += (int64_t) streamed[k].size();
        if (!light.empty() && im.lt_calls % (prof ? 40 : 400) == 0)
            std::fprintf(stderr, "xe tier: CPU-light (T %d): %lld calls, %lld experts on the pool (%.1f GB; %lld streamed), "
                         "quantise %.0f ms, pool %.0f ms (%.1f GB/s), sum %.0f ms, then the cards %.0f ms more\\n", light_t,
                         (long long) im.lt_calls, (long long) im.lt_experts, im.lt_bytes / 1e9, (long long) im.lt_streamed,
                         im.lt_q_ms, im.lt_run_ms, im.lt_run_ms > 0 ? im.lt_bytes / 1e6 / im.lt_run_ms : 0.0, im.lt_sum_ms,
                         im.lt_wait_ms);
    }
    for (size_t k = 0; k < NC; ++k) if (!ok[k]) return false;
    if (!light_ok) return false;
""")

# ---- combine: the CPU part as one more input
rep("""    // ---- combine: the other cards' parts copied card-to-card onto the dense card and added there
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    const size_t NE = (size_t) n * H;
    if (out_dev) {
        if (!dc) { std::fprintf(stderr, "xe tier: device output without a dense card\\n"); return false; }
        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        sycl::event cev;
""", """    // ---- combine: the other cards' parts copied card-to-card onto the dense card and added there
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (out_dev) {
        if (!dc) { std::fprintf(stderr, "xe tier: device output without a dense card\\n"); return false; }
        std::vector<const XeCard*> others;
        for (auto& cp : im.cards) if (cp->busy && !cp->dense) others.push_back(cp.get());
        sycl::event cev;
        const float* cpu_part = nullptr;   // STRATA_XE_CPU_LIGHT: the pool's sum, copied onto the dense card first
        if (!light.empty()) {
            std::lock_guard<PrioMutex> lk(g_submit);
            if (!grow_dev(im.lt_dev, im.lt_dev_cap, NE, dc->q)) {
                std::fprintf(stderr, "xe tier: dense card: CPU-light buffer allocation failed\\n");
                return false;
            }
            dc->q.memcpy(im.lt_dev, im.lt_hs, NE * 4);
            cpu_part = im.lt_dev;
        }
""")
rep("""            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            if (cmb == 2) {
                const sycl::half* parts = dc->cX2h;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += (float) parts[(size_t) idx[k] * NE + i];
                    o[i] = sm;
                });
            } else {
                const float* parts = dc->cX2;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += parts[(size_t) idx[k] * NE + i];
                    o[i] = sm;
                });
            }
""", """            const float* own = dc->busy ? dc->cOUT : nullptr;
            float* o = (float*) out_dev;
            if (cmb == 2) {
                const sycl::half* parts = dc->cX2h;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += (float) parts[(size_t) idx[k] * NE + i];
                    if (cpu_part) sm += cpu_part[i];
                    o[i] = sm;
                });
            } else {
                const float* parts = dc->cX2;
                cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                    float sm = own ? own[i] : 0.0f;
                    for (int k = 0; k < np; ++k) sm += parts[(size_t) idx[k] * NE + i];
                    if (cpu_part) sm += cpu_part[i];
                    o[i] = sm;
                });
            }
""")
rep("""            cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                float s = own ? own[i] : 0.0f;
                for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
                o[i] = s;
            });
""", """            cev = dc->q.parallel_for(sycl::range<1>(NE), [=](sycl::id<1> i) {
                float s = own ? own[i] : 0.0f;
                for (int k = 0; k < np; ++k) s += parts[(size_t) k * NE + i];
                if (cpu_part) s += cpu_part[i];
                o[i] = s;
            });
""")
rep("""        if (first) std::memset(out_host, 0, NE * 4);
    }
""", """        if (first) std::memset(out_host, 0, NE * 4);
        if (!light.empty()) for (size_t i = 0; i < NE; ++i) out_host[i] += im.lt_hs[i];
    }
""")

# ---- close (the dense card's buffer before the cards go: im.cards is cleared below)
rep("""    im.heads.clear();
    for (auto& cp : im.cards) {
""", """    im.heads.clear();
    if (im.lt_dev && im.dense >= 0 && (size_t) im.dense < im.cards.size()) {
        sycl::queue& q = im.cards[(size_t) im.dense]->q;
        q.wait();
        sycl::free(im.lt_dev, q);
    }
    im.lt_dev = nullptr;
    im.lt_dev_cap = 0;
    for (auto& cp : im.cards) {
""")
rep("""    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
""", """    if (im.hsum) { im.hsum_ev.wait(); sycl::free(im.hsum, im.ctx); im.hsum = nullptr; }
    if (im.lt_hs) { sycl::free(im.lt_hs, im.ctx); im.lt_hs = nullptr; im.lt_hs_cap = 0; }
    if (im.lt_x) { sycl::free(im.lt_x, im.ctx); im.lt_x = nullptr; im.lt_x_cap = 0; }
""")
open(P, "w").write(s)
print("patched", P)
