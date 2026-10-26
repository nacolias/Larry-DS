#!/usr/bin/env python3
# patch-ltcompact.py [TIER_CPP] - larry, STRATA_XE_LT_COMPACT (default 1; 0 = the old full-width form): a prompt call's
# CPU-light experts (STRATA_XE_CPU_LIGHT, <= 4 tokens each) touch a few dozen of the call's rows, but their X was read
# back off the dense card whole (n x 20 KB: 84 MB a 4096-token call, on card 0's link beside the expert cards' X reads)
# and their weighted sum went to the combine as a full n x n_embd array that the combine kernel read from pinned host
# memory over PCIe - STRATA_GPU_PTL (abwT62): a fresh 8K prompt's combines took 680 ms of card 0's queue (8.5 ms each,
# 0.4-0.9 ms at 12K / 16K where no call runs light experts).  Now the call's light rows are listed once: card 0
# gathers just those X rows (a kernel on its copy queue) and reads them back, the pool's sum fills one compact row per
# token, and after the combine a scatter kernel adds those rows (the same additions in the same order: bit-identical).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_LT_COMPACT" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    float* lt_x = nullptr;                 ///< pinned: a pipelined call's X read back off the dense card
    size_t lt_x_cap = 0;
""", """    float* lt_x = nullptr;                 ///< pinned: a pipelined call's X read back off the dense card
    size_t lt_x_cap = 0;
    // larry, STRATA_XE_LT_COMPACT (patch-ltcompact.py): the call's light rows only
    std::vector<int32_t> lt_slot;          ///< [token]: its compact row, -1
    int32_t* lt_rows = nullptr;            ///< pinned: compact row -> token (read by card 0's gather / scatter kernels)
    size_t lt_rows_cap = 0;
    float* lt_xg = nullptr;                ///< the dense card: the gathered X rows (read back into lt_x)
    size_t lt_xg_cap = 0;
    int lt_nrows = -1;                     ///< this call's compact rows (-1: lt_hs is full width)
""")
# the call's light rows, before X is read back
rep("""            if (!im.lt_hs) { std::fprintf(stderr, "xe tier: CPU-light sum buffer allocation failed\\n"); light_ok = false; }
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
""", """            if (!im.lt_hs) { std::fprintf(stderr, "xe tier: CPU-light sum buffer allocation failed\\n"); light_ok = false; }
            // larry, STRATA_XE_LT_COMPACT: the tokens the light experts serve, a compact row each (first-seen order)
            static const int lt_compact = env_i("STRATA_XE_LT_COMPACT", 1);
            int nrows = 0;
            im.lt_nrows = -1;
            if (light_ok && lt_compact) {
                im.lt_slot.assign((size_t) n, -1);
                for (int32_t e : light)
                    for (int32_t sk : by_e[(size_t) e]) {
                        const int t = sk / K;
                        if (im.lt_slot[(size_t) t] < 0) im.lt_slot[(size_t) t] = nrows++;
                    }
                if (im.lt_rows_cap < (size_t) nrows) {
                    if (im.lt_rows) sycl::free(im.lt_rows, im.ctx);
                    const size_t cap = std::max<size_t>((size_t) nrows, 1024);
                    im.lt_rows = sycl::malloc_host<int32_t>(cap, im.ctx);
                    im.lt_rows_cap = im.lt_rows ? cap : 0;
                }
                if (!im.lt_rows) { std::fprintf(stderr, "xe tier: CPU-light row list allocation failed\\n"); light_ok = false; }
                else for (int t = 0; t < n; ++t) if (im.lt_slot[(size_t) t] >= 0) im.lt_rows[im.lt_slot[(size_t) t]] = t;
            }
            const bool xl_compact = light_ok && lt_compact && x_dev;
            const float* xl = x;
            if (light_ok && x_dev) {   // a pipelined call: X off the dense card (its host copy is stale)
                const size_t xn = xl_compact ? (size_t) nrows * H : NE;
                if (im.lt_x_cap < xn) {
                    if (im.lt_x) sycl::free(im.lt_x, im.ctx);
                    im.lt_x = sycl::malloc_host<float>(xn, im.ctx);
                    im.lt_x_cap = im.lt_x ? xn : 0;
                }
                XeCard& dcx = *im.cards[(size_t) im.dense];
                if (xl_compact && im.lt_xg_cap < xn) {   // the dense card's gather buffer (grown, never on the decode path)
                    std::lock_guard<PrioMutex> lk(g_submit);
                    if (im.lt_xg) { dcx.pq.wait(); sycl::free(im.lt_xg, dcx.pq); }
                    const size_t cap = std::max<size_t>(xn, (size_t) 256 * H);
                    im.lt_xg = sycl::malloc_device<float>(cap, dcx.pq);
                    im.lt_xg_cap = im.lt_xg ? cap : 0;
                }
                if (!im.lt_x || (xl_compact && !im.lt_xg)) { std::fprintf(stderr, "xe tier: CPU-light X buffer allocation failed\\n"); light_ok = false; }
                else {
                    sycl::event xev;
                    {
                        std::lock_guard<PrioMutex> lk(g_submit);
                        if (xl_compact) {   // larry, STRATA_XE_LT_COMPACT: just the light rows, gathered on the dense card
                            const int32_t* rr = im.lt_rows;
                            float* xg = im.lt_xg;
                            const float* xs = x_dev;
                            const size_t h4 = (size_t) H / 4;
                            dcx.pq.parallel_for(sycl::range<1>((size_t) nrows * h4), [=](sycl::id<1> i) {
                                const size_t j = i[0] / h4, c = i[0] % h4;
                                reinterpret_cast<sycl::float4*>(xg + j * h4 * 4)[c] =
                                    reinterpret_cast<const sycl::float4*>(xs + (size_t) rr[j] * h4 * 4)[c];
                            });
                            xev = dcx.pq.memcpy(im.lt_x, xg, xn * 4);
                        } else {
                            xev = dcx.pq.memcpy(im.lt_x, x_dev, NE * 4);
                        }
                    }
                    xev.wait();
                    xl = im.lt_x;
                }
            }
            auto xrow = [&](int t) -> const float* {   // larry, STRATA_XE_LT_COMPACT
                return xl_compact ? xl + (size_t) im.lt_slot[(size_t) t] * H : xl + (size_t) t * H;
            };
""")
rep("""                            if (fs < 0) cpu::act_quant_any(xl + (size_t) t * H, (int) H, im.lt_actq[(size_t) t]);
                            else cpu::native_quant_act(f, xl + (size_t) t * H, na);
""", """                            if (fs < 0) cpu::act_quant_any(xrow(t), (int) H, im.lt_actq[(size_t) t]);
                            else cpu::native_quant_act(f, xrow(t), na);
""")
rep("""                const double ts = now_ms();
                std::memset(im.lt_hs, 0, NE * sizeof(float));
                for (size_t i = 0; i < light.size(); ++i) {
                    const std::vector<int32_t>& sl = by_e[(size_t) light[i]];
                    for (size_t si = 0; si < sl.size(); ++si) {
                        const int t = sl[si] / K;
                        const float wt = w[(size_t) sl[si]];
                        float* o = im.lt_hs + (size_t) t * H;
""", """                const double ts = now_ms();
                const bool sc = lt_compact != 0;   // larry, STRATA_XE_LT_COMPACT: a compact row a token
                std::memset(im.lt_hs, 0, (sc ? (size_t) nrows * H : NE) * sizeof(float));
                for (size_t i = 0; i < light.size(); ++i) {
                    const std::vector<int32_t>& sl = by_e[(size_t) light[i]];
                    for (size_t si = 0; si < sl.size(); ++si) {
                        const int t = sl[si] / K;
                        const float wt = w[(size_t) sl[si]];
                        float* o = im.lt_hs + (size_t) (sc ? im.lt_slot[(size_t) t] : t) * H;
""")
rep("""                t_sum = now_ms() - ts;
            }
            ++im.lt_calls;
""", """                t_sum = now_ms() - ts;
                if (sc) im.lt_nrows = nrows;
            }
            ++im.lt_calls;
""")
# the combine: no full-width CPU part; its rows scattered in after
rep("""        const float* cpu_part = light.empty() ? nullptr : im.lt_hs;
""", """        const float* cpu_part = light.empty() || im.lt_nrows >= 0 ? nullptr : im.lt_hs;   // larry, LT_COMPACT: rows after
""")
rep("""                if (cpu_part) s += cpu_part[i];
                o[i] = s;
            });
        }
        cev.wait();
""", """                if (cpu_part) s += cpu_part[i];
                o[i] = s;
            });
        }
        if (!light.empty() && im.lt_nrows > 0) {   // larry, STRATA_XE_LT_COMPACT: the CPU part's rows (the sum's last add)
            std::lock_guard<PrioMutex> lk(g_submit);
            const int32_t* rr = im.lt_rows;
            const float* cp = im.lt_hs;
            float* o = (float*) out_dev;
            const size_t hh = (size_t) H;
            cev = dc->q.parallel_for(sycl::range<1>((size_t) im.lt_nrows * hh), [=](sycl::id<1> i) {
                const size_t j = i[0] / hh, h = i[0] % hh;
                o[(size_t) rr[j] * hh + h] += cp[i[0]];
            });
        }
        cev.wait();
""")
rep("""        if (!light.empty()) for (size_t i = 0; i < NE; ++i) out_host[i] += im.lt_hs[i];
""", """        if (!light.empty() && im.lt_nrows >= 0) {   // larry, STRATA_XE_LT_COMPACT
            for (int j = 0; j < im.lt_nrows; ++j) {
                float* o = out_host + (size_t) im.lt_rows[j] * H;
                const float* p = im.lt_hs + (size_t) j * H;
                for (int64_t h = 0; h < H; ++h) o[h] += p[h];
            }
        } else if (!light.empty()) for (size_t i = 0; i < NE; ++i) out_host[i] += im.lt_hs[i];
""")
rep("""    if (im.lt_x) { sycl::free(im.lt_x, im.ctx); im.lt_x = nullptr; im.lt_x_cap = 0; }
""", """    if (im.lt_x) { sycl::free(im.lt_x, im.ctx); im.lt_x = nullptr; im.lt_x_cap = 0; }
    if (im.lt_rows) { sycl::free(im.lt_rows, im.ctx); im.lt_rows = nullptr; im.lt_rows_cap = 0; }   // larry, LT_COMPACT
    if (im.lt_xg && im.dense >= 0 && (size_t) im.dense < im.cards.size()) {
        sycl::queue& q = im.cards[(size_t) im.dense]->pq;
        q.wait();
        sycl::free(im.lt_xg, q);
    }
    im.lt_xg = nullptr; im.lt_xg_cap = 0;
""")
open(P, "w").write(s)
print("patched", P)
