// tp_layer.cpp - larry, the new engine's step 0: can a DeepSeek-V4.1 decode layer (1 token) with the dense half split
// over the four B70s, run with the host out of the loop, take under ~300 us (today: 564 us a layer)?
//
// The layer as card 0 runs it today (unitrace logs/ut-utpd, plain decode: 52 kernels, 443 us of kernel time a layer),
// rebuilt from real parts where they carry the bytes and stand-ins where they do not:
//   * every mat-vec is real: fastmv (the engine's) on real-size random planar Q6_K / F16 weights, distinct per layer
//     (cold in the L2, as in decode): q_a+kv (QKV_FUSE), the compressor (+ its gate on ratio-2 layers), the indexer's
//     q_b / k / proj on indexer layers, q_b, wo_a (8 groups), wo_b, the router (F16), the shared expert; quant_x
//     before each Q6_K one;
//   * the ~35 small ops (norms, rope, hc mixes, gathers, copies, the indexer's top-k chain) are delay kernels of their
//     traced GPU durations; the attention core a delay (22.9 + 3.1 us for 64 heads; 6.5 + 3.1 for 16); routed experts
//     delays of 22 us each (test_ds41_xe --bench-q8: 3 experts 66 us at 1 token), their cards drawn per layer;
//   * cross-card sums are real card-to-card copies + a sum kernel; cross-card dependencies SYCL events (GPU-side waits).
// Layouts:
//   base  today's: the dense half on card 0, experts on cards 1-3 (13 / 29 / 29 / 29 % of the picks), the parts copied
//         into card 0's slots and summed there
//   tp4   the dense half split: q_b's rows by head (16 heads a card), attention by head, wo_a 2 groups a card, wo_b's
//         matching 2048 columns (a partial), the shared expert's 2304 by 768 / 512 / 512 / 512 (a partial), experts on
//         all four (25 % each); both partial sums all-reduced card to card; the rest replicated on every card
// Execution:
//   gated  all commands of a token (40 layers, four queues) queued behind a gate kernel, then the GPUs run them back to
//          back: the GPUs' own time with the host out of the loop (the upper bound of host-free eager submission)
//   paced  the same commands submitted live: the host in the loop (today's engine)
// Variants: unfused (as traced) and fused (each run of consecutive small ops one kernel of their summed time, quant_x
// folded into the mat-vec): what kernel fusion adds.  Micro: the GPU's start-to-start cost of a dependent kernel,
// eager vs SYCL graph replay, and a 4-card all-reduce of 5120 floats.
// Build (larry): icpx -fsycl -O3 -std=c++17 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
//                tp_layer.cpp fastmv.cpp -o tp_layer       Run: GPU clocks pinned, service stopped.  ./tp_layer [tokens]
#include "fastmv.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
namespace sx = sycl::ext::oneapi::experimental;

constexpr int NC = 4;         // cards
constexpr int64_t H = 5120;   // model width
constexpr int NL = 40;        // layers a token
bool is_index(int l) { return l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 || l == 32 || l == 36; }
int ratio(int l) { return l < 2 ? 0 : l < 20 ? 2 : 1; }
constexpr double EXPERT_US = 22.0;   // one routed expert, 1 token (gate/up + down)

// ---- weights
struct W {   // one weight on one card
    void* p = nullptr;
    int64_t N = 0, K = 0, G = 1;
    bool f16 = false;
    int64_t nb() const { return N * (K / 256); }
    size_t batch_bytes() const { return f16 ? (size_t) N * K * 2 : (size_t) nb() * 210; }
    size_t bytes() const { return batch_bytes() * G; }
};
// host source data per byte size (random planar Q6_K: ql | qh | scales | d planes; or f16 rows), uploaded per layer
std::map<size_t, std::vector<uint8_t>> g_src;
const std::vector<uint8_t>& src_for(const W& w) {
    auto it = g_src.find(w.bytes() * 4 + (w.f16 ? 1 : 0));
    if (it != g_src.end()) return it->second;
    std::vector<uint8_t> h(w.bytes());
    std::mt19937 rng((uint32_t) w.bytes());
    if (w.f16) {
        sycl::half* hp = (sycl::half*) h.data();
        for (size_t i = 0; i < h.size() / 2; ++i) hp[i] = (sycl::half) ((float) ((int) (rng() % 2001) - 1000) * 1e-5f);
    } else {
        for (int64_t g = 0; g < w.G; ++g) {
            uint8_t* b = h.data() + g * w.batch_bytes();
            const int64_t nb = w.nb();
            for (int64_t i = 0; i < nb * 192; ++i) b[i] = (uint8_t) (rng() & 0xFF);
            for (int64_t i = 192 * nb; i < 208 * nb; ++i) b[i] = (uint8_t) (rng() % 64);
            const sycl::half hd = (sycl::half) 0.0005f;
            for (int64_t i = 0; i < nb; ++i) std::memcpy(b + 208 * nb + 2 * i, &hd, 2);
        }
    }
    return g_src.emplace(w.bytes() * 4 + (w.f16 ? 1 : 0), std::move(h)).first->second;
}
W make_w(sycl::queue& q, int64_t N, int64_t K, int64_t G, bool f16) {
    W w; w.N = N; w.K = K; w.G = G; w.f16 = f16;
    w.p = sycl::malloc_device<uint8_t>(w.bytes(), q);
    q.memcpy(w.p, src_for(w).data(), w.bytes());
    return w;
}

struct LayerW {   // one card's weights of one layer (unused ones stay empty)
    W qkv, ckv, cgate, iqb, ik, iproj, qb, woa, wob, router, sgu, sdn;
};

// ---- per-card state
struct Card {
    explicit Card(sycl::queue qq) : q(std::move(qq)) {}
    sycl::queue q;
    std::vector<LayerW> base, tp;      // [layer]
    float* x = nullptr;                // activations in (garbage values are fine: timing only)
    float* y = nullptr;                // mat-vec outputs
    int8_t* xq = nullptr;
    float* xd = nullptr;
    float* part = nullptr;             // the partial sum to all-reduce
    float* slots = nullptr;            // [2 all-reduces][2 parities][NC][H] incoming partials
    float* scratch = nullptr;
    int64_t kernels = 0;               // commands queued (kernels + copies), for the per-layer counts
    // graph mode's cross-card sync (a recorded graph lives on one card: no events across cards inside it)
    int* counter = nullptr;            // tokens started (the first node of every token graph bumps it)
    int* seqsrc = nullptr;             // [MAXS] this token's sequence numbers, copied into the peers' flags
    int* flags = nullptr;              // [NC][MAXS] incoming flags: card k's sync point s
    int* err = nullptr;                // spins that gave up
};
struct Ctx {
    std::vector<Card> c;
    float* busy_buf = nullptr;
    int* hflag = nullptr;              // pinned: [0] gate end, [1 + k] card k's last command
    int64_t busy_iters = 1 << 22;
    double delay_iters_per_us = 100;
    int tok = 0;
    sycl::event gate;
    bool gated = false;
    bool fused = false;
    bool graph = false;                // record each card's token as a SYCL graph; cross-card sync by flags
    int sync_idx = 0;                  // graph mode: the next sync point of this token
};
constexpr int MAXS = 128;              // sync points a token (2 a layer)
constexpr int64_t SPIN_MAX = 1 << 20;  // polls before a waiting card gives up (and counts an error)

sycl::event delay(Ctx& C, int k, double us, const std::vector<sycl::event>& deps = {}) {
    const int64_t it = std::max<int64_t>(1, (int64_t) std::llround(us * C.delay_iters_per_us));
    float* p = C.c[k].scratch;
    ++C.c[k].kernels;
    return C.c[k].q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> i) {
            float v = 1.0f + (float) i.get_local_id(0);
            for (int64_t j = 0; j < it; ++j) v = v * 0.9999999f + 1e-7f;
            if (v == 12345.678f) p[0] = v;
        });
    });
}
// a run of small ops: one kernel each, or (fused) one kernel of their summed time
void small(Ctx& C, int k, std::initializer_list<double> us) {
    if (C.fused) {
        const double t = std::accumulate(us.begin(), us.end(), 0.0);
        if (t > 0) delay(C, k, t);
    } else {
        for (double u : us) delay(C, k, u);
    }
}
void qx(Ctx& C, int k, int64_t K, int64_t G = 1) {
    if (C.fused) return;
    Card& c = C.c[k];
    ++c.kernels;
    fastmv::quant_x(c.q, c.x, K, K, 1, G, K, c.xq, c.xd);
}
sycl::event mv(Ctx& C, int k, const W& w, float* y = nullptr) {
    Card& c = C.c[k];
    ++c.kernels;
    fastmv::Args a;
    a.w = w.p; a.K = w.K; a.N = w.N; a.G = w.G; a.g_div = 1;
    a.w_batch_bytes = w.batch_bytes();
    a.x = c.x; a.x_col_stride = w.K; a.x_batch_stride = w.K;
    a.y = y ? y : c.y; a.y_col_stride = w.N * w.G; a.y_batch_stride = w.N; a.ncols = 1;
    if (w.f16) {
        a.w_row_bytes = (size_t) w.K * 2;
    } else {
        a.n_blocks_total = w.nb();
        a.xq = c.xq; a.xd = c.xd;
    }
    if (!fastmv::launch(c.q, w.f16 ? fastmv::WType::F16 : fastmv::WType::Q6K_R8, a)) {
        std::fprintf(stderr, "fastmv: shape not covered: N %lld K %lld G %lld f16 %d\n", (long long) w.N, (long long) w.K,
                     (long long) w.G, (int) w.f16);
        std::exit(1);
    }
    return {};
}
sycl::event sum_kernel(Ctx& C, int k, float* y, const float* s, int parts, const std::vector<sycl::event>& deps) {
    ++C.c[k].kernels;
    return C.c[k].q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> i) {
            float v = y[i];
            for (int p = 0; p < parts; ++p) v += s[p * H + i];
            y[i] = v;
        });
    });
}
// ---- graph mode: sync point s of token t carries t * 4096 + s; the sender copies it into each waiting card's flag
// slot right after the data copies (the same in-order queue: the data lands first); the waiting card spins on its own
// memory, bounded (the tier's release mechanism: a flag in device memory written by a copy on another card's queue)
void g_begin(Ctx& C, int k) {
    int* ctr = C.c[k].counter;
    ++C.c[k].kernels;
    C.c[k].q.single_task([=] { *ctr += 1; });
}
void g_signal(Ctx& C, int k, int s, const std::vector<int>& to) {
    Card& c = C.c[k];
    int* ctr = c.counter;
    int* src = c.seqsrc + s;
    ++c.kernels;
    c.q.single_task([=] { *src = *ctr * 4096 + s; });
    for (int j : to) {
        c.q.memcpy(C.c[j].flags + k * MAXS + s, src, 4);
        ++c.kernels;
    }
}
void g_wait(Ctx& C, int j, int s, const std::vector<int>& from) {
    Card& c = C.c[j];
    int* ctr = c.counter;
    int* fl = c.flags;
    int* err = c.err;
    int kk[NC] = {0, 0, 0, 0};
    const int nk = (int) from.size();
    for (int i = 0; i < nk; ++i) kk[i] = from[i];
    const int k0 = kk[0], k1 = kk[1], k2 = kk[2];
    ++c.kernels;
    c.q.single_task([=] {
        const int want = *ctr * 4096 + s;
        const int ks[3] = {k0, k1, k2};
        for (int i = 0; i < nk; ++i) {
            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>
                f(fl[ks[i] * MAXS + s]);
            int64_t n = 0;
            while (f.load() != want && ++n < SPIN_MAX) {}
            if (n >= SPIN_MAX) err[0] += 1;
        }
    });
}
std::vector<int> others(int j) { std::vector<int> v; for (int k = 0; k < NC; ++k) if (k != j) v.push_back(k); return v; }

// every card holds a partial in c.part: send it to the three others' slots, then sum (all four end with the total)
void allreduce(Ctx& C, int ar, int parity) {
    if (C.graph) {
        const int s = C.sync_idx++;
        for (int k = 0; k < NC; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (j == k) continue;
                C.c[k].q.memcpy(C.c[j].slots + ((size_t) (ar * 2 + parity) * NC + k) * H, C.c[k].part, H * 4);
                ++C.c[k].kernels;
            }
            g_signal(C, k, s, others(k));
        }
        for (int j = 0; j < NC; ++j) {
            g_wait(C, j, s, others(j));
            sum_kernel(C, j, C.c[j].part, C.c[j].slots + (size_t) (ar * 2 + parity) * NC * H, NC, {});
        }
        return;
    }
    std::vector<std::vector<sycl::event>> in(NC);
    for (int k = 0; k < NC; ++k)
        for (int j = 0; j < NC; ++j) {
            if (j == k) continue;
            float* dst = C.c[j].slots + ((size_t) (ar * 2 + parity) * NC + k) * H;
            in[j].push_back(C.c[k].q.memcpy(dst, C.c[k].part, H * 4));
            ++C.c[k].kernels;
        }
    for (int j = 0; j < NC; ++j) {
        // the slot of card j itself stays zero: sum all NC (one wasted read of 20 KB keeps the kernel simple)
        sum_kernel(C, j, C.c[j].part, C.c[j].slots + (size_t) (ar * 2 + parity) * NC * H, NC, in[j]);
    }
}

// ---- the layer programs
const std::initializer_list<double> INDEX_CHAIN = {0.6, 1.9, 1.8, 0.6, 0.6, 1.6, 5.0, 1.2, 2.1, 1.9, 0.7, 0.6,
                                                   0.6, 0.6, 1.4, 8.0, 1.3, 2.7, 1.3, 1.2, 5.2, 2.0, 2.3};
void attention_pre(Ctx& C, int k, const LayerW& w, int l) {   // replicated in both layouts
    small(C, k, {4.4, 3.8});                       // hc_fa, hc_fb
    qx(C, k, H);
    mv(C, k, w.qkv);                               // q_a + kv
    small(C, k, {2.3});                            // q_a norm
    if (is_index(l)) mv(C, k, w.iqb);              // indexer q_b (F16)
    small(C, k, {2.8, 0.8, 5.5});
    if (ratio(l) > 0) mv(C, k, w.ckv);             // compressor kv (F16)
    if (ratio(l) == 2) mv(C, k, w.cgate);          // compressor gate (F16)
    small(C, k, {2.0});
    if (is_index(l)) mv(C, k, w.ik);               // indexer k (F16)
    small(C, k, {1.5, 1.5, 1.4, 4.8, 2.4});
    if (is_index(l)) {
        mv(C, k, w.iproj);                         // indexer weights (F16)
        if (C.fused) small(C, k, {std::accumulate(INDEX_CHAIN.begin(), INDEX_CHAIN.end(), 0.0)});
        else for (double u : INDEX_CHAIN) delay(C, k, u);
    }
}
void attention(Ctx& C, int k, const LayerW& w, bool split) {
    qx(C, k, 1280);
    mv(C, k, w.qb);                                                    // q_b (all heads, or this card's 16)
    small(C, k, {3.1, 2.0, 1.5, 1.3, 5.3, 1.5, 1.9, 1.4, 3.9, 2.4, 4.0, 1.3, 0.9, 1.2});
    delay(C, k, split ? 6.5 : 22.9);                                   // the attention core
    delay(C, k, 3.1);                                                  // its combine
    small(C, k, {2.3});                                                // inverse rope of o
    qx(C, k, 4096, w.woa.G);
    mv(C, k, w.woa);                                                   // wo_a (8 or 2 groups)
    qx(C, k, w.wob.K);
    mv(C, k, w.wob, split ? C.c[k].part : nullptr);                    // wo_b (whole, or a partial)
}
void ffn_pre(Ctx& C, int k, const LayerW& w) {
    small(C, k, {6.4, 4.1});                                           // hc_fa / hc_fb of the ffn
    mv(C, k, w.router);                                                // router (F16)
    small(C, k, {7.3, 1.8, 1.8, 0.8, 2.2, 1.7, 1.2, 1.2});             // top-k, job prep, readback copies
}
void shexp(Ctx& C, int k, const LayerW& w, bool split) {
    qx(C, k, H);
    mv(C, k, w.sgu);                                                   // gate | up
    small(C, k, {1.5});                                                // swiglu
    qx(C, k, w.sdn.K);
    mv(C, k, w.sdn, split ? C.c[k].part : nullptr);                    // down (whole, or a partial)
}
void experts(Ctx& C, int k, int n, const std::vector<sycl::event>& deps = {}) {
    if (n <= 0) {
        if (!deps.empty()) delay(C, k, 0.5, deps);
        return;
    }
    delay(C, k, n * EXPERT_US * 0.62, deps);                           // gate/up + swiglu
    delay(C, k, n * EXPERT_US * 0.38);                                 // down + weighted sum
}

// the routed experts' cards for one layer of one token (6 picks)
std::vector<int> draw(std::mt19937& rng, const double* p) {
    std::vector<int> n(NC, 0);
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < 6; ++i) {
        double r = u(rng), a = 0;
        int k = 0;
        for (; k < NC - 1; ++k) { a += p[k]; if (r < a) break; }
        ++n[k];
    }
    return n;
}

void token_base(Ctx& C, std::mt19937& rng) {
    static const double P[NC] = {0.13, 0.29, 0.29, 0.29};
    for (int l = 0; l < NL; ++l) {
        const LayerW& w = C.c[0].base[l];
        attention_pre(C, 0, w, l);
        attention(C, 0, w, false);
        ffn_pre(C, 0, w);
        const std::vector<int> n = draw(rng, P);
        std::vector<sycl::event> parts;
        if (C.graph) {
            const int s1 = C.sync_idx++, s2 = C.sync_idx++;
            g_signal(C, 0, s1, others(0));                             // the routing is known
            for (int k = 1; k < NC; ++k) {
                g_wait(C, k, s1, {0});
                experts(C, k, n[k]);
                C.c[k].q.memcpy(C.c[0].slots + (size_t) (l % 2) * NC * H + k * H, C.c[k].part, H * 4);
                ++C.c[k].kernels;
                g_signal(C, k, s2, {0});
            }
            shexp(C, 0, w, false);
            experts(C, 0, n[0]);
            g_wait(C, 0, s2, others(0));
            sum_kernel(C, 0, C.c[0].y, C.c[0].slots + (size_t) (l % 2) * NC * H, NC, {});
        } else {
            const sycl::event routed = delay(C, 0, 0.5);               // the routing is known
            for (int k = 1; k < NC; ++k) {                             // cards 1-3: their experts, then the part to card 0
                experts(C, k, n[k], {routed});
                parts.push_back(C.c[k].q.memcpy(C.c[0].slots + (size_t) (l % 2) * NC * H + k * H, C.c[k].part, H * 4));
                ++C.c[k].kernels;
            }
            shexp(C, 0, w, false);
            experts(C, 0, n[0]);
            sum_kernel(C, 0, C.c[0].y, C.c[0].slots + (size_t) (l % 2) * NC * H, NC, parts);
        }
        small(C, 0, {1.8});                                            // hc_post2 (finish)
    }
}
void token_tp(Ctx& C, std::mt19937& rng) {
    static const double P[NC] = {0.25, 0.25, 0.25, 0.25};
    for (int l = 0; l < NL; ++l) {
        for (int k = 0; k < NC; ++k) {
            const LayerW& w = C.c[k].tp[l];
            attention_pre(C, k, w, l);
            attention(C, k, w, true);
        }
        allreduce(C, 0, l % 2);                                        // the attention output
        const std::vector<int> n = draw(rng, P);
        for (int k = 0; k < NC; ++k) {
            const LayerW& w = C.c[k].tp[l];
            ffn_pre(C, k, w);
            shexp(C, k, w, true);
            experts(C, k, n[k]);
        }
        allreduce(C, 1, l % 2);                                        // shared expert + routed experts
        for (int k = 0; k < NC; ++k) small(C, k, {1.8});
    }
}

// ---- timing
sycl::event busy(Ctx& C, int64_t iters, int tok) {
    float* p = C.busy_buf;
    volatile int* hf = C.hflag;
    return C.c[0].q.single_task([=] {
        float x = p[0];
        for (int64_t i = 0; i < iters; ++i) x = x * 0.9999999f + 1e-7f;
        p[1] = x;
        if (tok) hf[0] = tok;
    });
}
void finish_marks(Ctx& C, int tok) {
    for (int k = 0; k < NC; ++k) {
        volatile int* hf = C.hflag;
        C.c[k].q.single_task([=] { hf[1 + k] = tok; });
    }
}
void wait_all(Ctx& C) { for (auto& c : C.c) c.q.wait(); }
// one token's commands queued behind the gate; returns the GPUs' us from the gate's end to the last card's end
double run_gated(Ctx& C, const std::function<void()>& token) {
    wait_all(C);
    const int tok = ++C.tok;
    C.gate = busy(C, C.busy_iters, tok);
    for (int k = 1; k < NC; ++k) delay(C, k, 0.5, {C.gate});          // cards 1-3 start after the gate too
    const auto t0 = clk::now();
    token();
    finish_marks(C, tok);
    const double sub = us_since(t0);
    volatile int* hf = C.hflag;
    if (hf[0] == tok) std::printf("    (warning: the gate opened while the host was still queueing: %.0f us of submission)\n", sub);
    while (hf[0] != tok) {}
    const auto t1 = clk::now();
    for (int k = 0; k < NC; ++k) while (hf[1 + k] != tok) {}
    const double gpu = us_since(t1);
    wait_all(C);
    return gpu;
}
double run_paced(Ctx& C, const std::function<void()>& token) {
    wait_all(C);
    const int tok = ++C.tok;
    const auto t0 = clk::now();
    token();
    finish_marks(C, tok);
    volatile int* hf = C.hflag;
    for (int k = 0; k < NC; ++k) while (hf[1 + k] != tok) {}
    const double us = us_since(t0);
    wait_all(C);
    return us;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
// graph mode: each card's token recorded once as a SYCL graph (cross-card sync by flags), then replayed: four graph
// launches a token from the host.  Returns the median wall us a token over `reps` replays; *spins = spins that gave up
double run_graph(Ctx& C, const std::function<void()>& token, int reps, int* spins) {
    wait_all(C);
    std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
    for (int k = 0; k < NC; ++k) g.emplace_back(C.c[k].q.get_context(), C.c[k].q.get_device());
    for (int k = 0; k < NC; ++k) g[k].begin_recording(C.c[k].q);
    C.sync_idx = 0;
    for (int k = 0; k < NC; ++k) g_begin(C, k);
    token();
    for (int k = 0; k < NC; ++k) g[k].end_recording(C.c[k].q);
    std::vector<sx::command_graph<sx::graph_state::executable>> e;
    for (int k = 0; k < NC; ++k) e.push_back(g[k].finalize());
    for (auto& c : C.c) c.q.memset(c.err, 0, 4);
    wait_all(C);
    std::vector<double> t;
    for (int r = 0; r < reps + 1; ++r) {
        const auto t0 = clk::now();
        for (int k = 0; k < NC; ++k) C.c[k].q.ext_oneapi_graph(e[k]);
        wait_all(C);
        if (r > 0) t.push_back(us_since(t0));   // the first replay warms up
    }
    *spins = 0;
    for (auto& c : C.c) { int v = 0; c.q.memcpy(&v, c.err, 4).wait(); *spins += v; }
    return median(t);
}

// ---- micro: GPU start-to-start of dependent tiny kernels, eager (queued behind the gate) vs a replayed SYCL graph
struct Launch { double eager = 0, graph = 0; };
Launch micro_launch(Ctx& C) {
    const int n = 200;
    auto chain = [&]() { for (int i = 0; i < n; ++i) delay(C, 0, 1.0); };
    std::vector<double> e;
    for (int r = 0; r < 5; ++r) e.push_back(run_gated(C, chain));
    // the same chain recorded as a graph on card 0 and replayed
    sycl::queue& q = C.c[0].q;
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    chain();
    g.end_recording(q);
    auto ge = g.finalize();
    std::vector<double> gr;
    for (int r = 0; r < 5; ++r) {
        gr.push_back(run_gated(C, [&]() { q.ext_oneapi_graph(ge); }));
    }
    double only = 0;
    {   // the delay kernel's own time (one long-running instance / its nominal us): the calibration check
        const auto t = clk::now();
        delay(C, 0, 2000.0).wait();
        only = us_since(t);
    }
    std::printf("micro: %d dependent 1-us kernels: eager %.2f us each, graph replay %.2f us each (a 2000-us delay took %.0f us)\n",
                n, median(e) / n, median(gr) / n, only);
    return {median(e) / n, median(gr) / n};
}
void micro_allreduce(Ctx& C) {
    const int n = 50;
    std::vector<double> t;
    for (int r = 0; r < 5; ++r) t.push_back(run_gated(C, [&]() { for (int i = 0; i < n; ++i) allreduce(C, i % 2, (i / 2) % 2); }));
    std::printf("micro: all-reduce of %lld floats over %d cards (3 copies out + a sum a card): %.1f us each\n", (long long) H, NC,
                median(t) / n);
}
}  // namespace

int main(int argc, char** argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 5;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Ctx C;
    std::vector<sycl::device> dev;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        dev = p.get_devices(sycl::info::device_type::gpu);
        if (!dev.empty()) break;
    }
    if (dev.size() < (size_t) NC) { std::printf("need %d Level Zero GPUs\n", NC); return 1; }
    for (int k = 0; k < NC; ++k) C.c.push_back(Card{sycl::queue(dev[k], sycl::property::queue::in_order())});
    std::printf("contexts equal: %d\n", (int) (C.c[0].q.get_context() == C.c[1].q.get_context()));
    C.hflag = sycl::malloc_host<int>(8, C.c[0].q);
    std::memset(C.hflag, 0, 8 * sizeof(int));
    C.busy_buf = sycl::malloc_device<float>(2, C.c[0].q);
    C.c[0].q.memset(C.busy_buf, 0, 8).wait();
    for (auto& c : C.c) {
        c.x = sycl::malloc_device<float>(8 * 4096, c.q);
        c.y = sycl::malloc_device<float>(32768 * 2, c.q);
        c.xq = sycl::malloc_device<int8_t>(8 * 4096, c.q);
        c.xd = sycl::malloc_device<float>(8 * 4096 / 32, c.q);
        c.part = sycl::malloc_device<float>(H, c.q);
        c.slots = sycl::malloc_device<float>(2 * 2 * NC * H, c.q);
        c.scratch = sycl::malloc_device<float>(16, c.q);
        c.counter = sycl::malloc_device<int>(1, c.q);
        c.seqsrc = sycl::malloc_device<int>(MAXS, c.q);
        c.flags = sycl::malloc_device<int>(NC * MAXS, c.q);
        c.err = sycl::malloc_device<int>(1, c.q);
        c.q.memset(c.counter, 0, 4); c.q.memset(c.seqsrc, 0, MAXS * 4); c.q.memset(c.flags, 0, NC * MAXS * 4);
        c.q.memset(c.err, 0, 4);
        c.q.memset(c.x, 0, 8 * 4096 * 4); c.q.memset(c.y, 0, 32768 * 2 * 4); c.q.memset(c.xq, 0, 8 * 4096);
        c.q.memset(c.xd, 0, 8 * 4096 / 32 * 4); c.q.memset(c.part, 0, H * 4); c.q.memset(c.slots, 0, 2 * 2 * NC * H * 4);
        c.q.wait();
    }
    {   // calibrate the gate (~250 ms) and the delay kernel
        busy(C, 1 << 20, 0).wait();
        auto t = clk::now();
        busy(C, 1 << 22, 0).wait();
        C.busy_iters = (int64_t) ((double) (1 << 22) * 250000.0 / std::max(us_since(t), 1.0));
        C.delay_iters_per_us = 100;
        delay(C, 0, 1000.0).wait();
        t = clk::now();
        delay(C, 0, 20000.0).wait();
        const double us = us_since(t);
        C.delay_iters_per_us = 100.0 * 20000.0 / us;
        std::printf("gate %lld iterations (~250 ms); delay kernel %.1f iterations/us\n", (long long) C.busy_iters, C.delay_iters_per_us);
    }
    // weights: every layer its own (cold), both layouts
    const auto tw = clk::now();
    const int64_t SH[NC] = {768, 512, 512, 512};
    for (int l = 0; l < NL; ++l) {
        LayerW b;
        sycl::queue& q0 = C.c[0].q;
        b.qkv = make_w(q0, 1792, H, 1, false);
        if (ratio(l) > 0) b.ckv = make_w(q0, 512, H, 1, true);
        if (ratio(l) == 2) b.cgate = make_w(q0, 512, H, 1, true);
        if (is_index(l)) { b.iqb = make_w(q0, 4096, 1280, 1, true); b.ik = make_w(q0, 128, 512, 1, true); b.iproj = make_w(q0, 32, H, 1, true); }
        b.qb = make_w(q0, 32768, 1280, 1, false);
        b.woa = make_w(q0, 1024, 4096, 8, false);
        b.wob = make_w(q0, H, 8192, 1, false);
        b.router = make_w(q0, 384, H, 1, true);
        b.sgu = make_w(q0, 4608, H, 1, false);
        b.sdn = make_w(q0, H, 2304, 1, false);
        C.c[0].base.push_back(b);
        for (int k = 0; k < NC; ++k) {
            sycl::queue& q = C.c[k].q;
            LayerW t;
            t.qkv = make_w(q, 1792, H, 1, false);
            if (ratio(l) > 0) t.ckv = make_w(q, 512, H, 1, true);
            if (ratio(l) == 2) t.cgate = make_w(q, 512, H, 1, true);
            if (is_index(l)) { t.iqb = make_w(q, 4096, 1280, 1, true); t.ik = make_w(q, 128, 512, 1, true); t.iproj = make_w(q, 32, H, 1, true); }
            t.qb = make_w(q, 8192, 1280, 1, false);
            t.woa = make_w(q, 1024, 4096, 2, false);
            t.wob = make_w(q, H, 2048, 1, false);
            t.router = make_w(q, 384, H, 1, true);
            t.sgu = make_w(q, 2 * SH[k], H, 1, false);
            t.sdn = make_w(q, H, SH[k], 1, false);
            C.c[k].tp.push_back(t);
        }
    }
    wait_all(C);
    g_src.clear();
    std::printf("weights up in %.1f s\n", us_since(tw) / 1e6);

    const Launch lc = micro_launch(C);
    micro_allreduce(C);

    std::mt19937 rng(20261022);
    struct Mode { const char* name; bool tp, fused; };
    for (const Mode& m : {Mode{"base  unfused", false, false}, Mode{"tp4   unfused", true, false},
                          Mode{"base  fused  ", false, true}, Mode{"tp4   fused  ", true, true}}) {
        C.fused = m.fused;
        auto token = [&]() { if (m.tp) token_tp(C, rng); else token_base(C, rng); };
        run_gated(C, token);   // warm-up
        const int64_t k0 = C.c[0].kernels;
        run_gated(C, token);
        const double per_layer = (double) (C.c[0].kernels - k0 - 1) / NL;   // card 0's commands a layer
        std::vector<double> g, p;
        for (int t = 0; t < tokens; ++t) g.push_back(run_gated(C, token));
        for (int t = 0; t < std::max(2, tokens / 2); ++t) p.push_back(run_paced(C, token));
        std::printf("%s: host-free %6.1f us a layer (%5.2f ms a token) | host-paced %6.1f us a layer | %4.1f commands a layer on card 0"
                    " -> graph replay ~%6.1f us a layer\n", m.name, median(g) / NL, median(g) / 1000, median(p) / NL, per_layer,
                    median(g) / NL - per_layer * (lc.eager - lc.graph));
    }
    for (const Mode& m : {Mode{"base  unfused", false, false}, Mode{"tp4   unfused", true, false},
                          Mode{"base  fused  ", false, true}, Mode{"tp4   fused  ", true, true}}) {
        C.fused = m.fused;
        C.graph = true;
        auto token = [&]() { if (m.tp) token_tp(C, rng); else token_base(C, rng); };
        int spins = 0;
        double us = 0;
        try {
            us = run_graph(C, token, std::max(3, tokens), &spins);
        } catch (const sycl::exception& ex) {
            std::printf("%s graph: SYCL exception: %s\n", m.name, ex.what());
            C.graph = false;
            continue;
        }
        C.graph = false;
        std::printf("%s graph replay: %6.1f us a layer (%5.2f ms a token, four graph launches a token)%s\n", m.name, us / NL,
                    us / 1000, spins ? "  ** spins gave up: timing invalid **" : "");
    }
    return 0;
}
