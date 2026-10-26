// tp_probe.cpp - larry: what splitting DeepSeek-V4.1's dense decode mat-vecs over the four B70s costs ON THE GPUs.
// The host's submissions are taken out: a gate kernel holds card 0 while N iterations are queued, then the host opens
// it and times the GPUs running them back to back (each iteration ordered after the previous one).
//
//   rt      a bare round trip: card 0 kernel -> cards 1-3 copy 8 KB from card 0, tiny kernel, copy 20 KB into card 0's
//           slots -> card 0 sums (the sync overhead of one scatter + one reduce)
//   oproj   the attention output: quant_x(o) -> wo_a (8 groups 4096->1024) -> quant_x -> wo_b (8192->5120)
//           card 0 alone vs head-parallel (card k: groups 2k, 2k+1, wo_b's matching 2048 columns, a partial sum)
//   qb      q_b (1280->32768) on card 0 vs row-split; the q heads stay on their card (head-parallel attention)
//   shexp   gate|up (5120->4608) -> swiglu -> down (2304->5120): card 0 alone vs cards 1-3 (768 of 2304 each)
//
// Each mode also runs host-paced (no gate: submit and go, as the engine does) for comparison.
// Build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" tp_probe.cpp fastmv.cpp
#include "fastmv.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }

struct Ctx {
    std::vector<sycl::device> dev;
    std::vector<sycl::queue> q;
    // the gate: a busy kernel on card 0 (no flags - a GPU polling host memory does not see host writes, and a release
    // copy on another queue can sit behind that card's blocked commands) that outlasts the host's submissions
    float* busy_buf = nullptr;
    int64_t busy_iters = 1 << 22;
    sycl::event busy_ev;
    int* hflag = nullptr;   // pinned: [0] the busy kernel ended, [1] the last iteration ended
    int tok = 0;
};

// planar Q6_K (R8 kernels): ql 128 | qh 64 | sc 16 | d 2 bytes per 256-weight block, each plane contiguous
struct Planar {
    int64_t N = 0, K = 0, G = 1;
    std::vector<uint8_t> h;   // G plane sets
    int64_t nb() const { return N * (K / 256); }
    size_t batch_bytes() const { return (size_t) nb() * 210; }
};
Planar make_planar(int64_t N, int64_t K, int64_t G, uint32_t seed) {
    Planar p; p.N = N; p.K = K; p.G = G;
    p.h.resize(p.batch_bytes() * G);
    std::mt19937 rng(seed);
    const int64_t nb = p.nb();
    for (int64_t g = 0; g < G; ++g) {
        uint8_t* b = p.h.data() + g * p.batch_bytes();
        for (int64_t i = 0; i < nb * 192; ++i) b[i] = (uint8_t) (rng() & 0xFF);
        for (int64_t i = 192 * nb; i < 208 * nb; ++i) b[i] = (uint8_t) (rng() % 64);
        const sycl::half hd = (sycl::half) 0.0005f;
        for (int64_t i = 0; i < nb; ++i) std::memcpy(b + 208 * nb + 2 * i, &hd, 2);
    }
    return p;
}
// rows [r0, r0 + n) of every batch
Planar rows_of(const Planar& s, int64_t r0, int64_t n) {
    Planar p; p.N = n; p.K = s.K; p.G = s.G;
    p.h.resize(p.batch_bytes() * p.G);
    const int64_t bpr = s.K / 256, snb = s.nb(), nb = p.nb();
    const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};
    for (int64_t g = 0; g < s.G; ++g)
        for (int k = 0; k < 4; ++k)
            std::memcpy(p.h.data() + g * p.batch_bytes() + po[k] * nb, s.h.data() + g * s.batch_bytes() + po[k] * snb + pb[k] * r0 * bpr,
                        pb[k] * nb);
    return p;
}
// columns [c0, c0 + n) (multiples of 256) of a one-batch matrix
Planar cols_of(const Planar& s, int64_t c0, int64_t n) {
    Planar p; p.N = s.N; p.K = n; p.G = 1;
    p.h.resize(p.batch_bytes());
    const int64_t sbpr = s.K / 256, bpr = n / 256, b0 = c0 / 256, snb = s.nb(), nb = p.nb();
    const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};
    for (int64_t r = 0; r < s.N; ++r)
        for (int64_t b = 0; b < bpr; ++b)
            for (int k = 0; k < 4; ++k)
                std::memcpy(p.h.data() + po[k] * nb + pb[k] * (r * bpr + b), s.h.data() + po[k] * snb + pb[k] * (r * sbpr + b0 + b), pb[k]);
    return p;
}
// batches [g0, g0 + n)
Planar batches_of(const Planar& s, int64_t g0, int64_t n) {
    Planar p; p.N = s.N; p.K = s.K; p.G = n;
    p.h.assign(s.h.begin() + g0 * s.batch_bytes(), s.h.begin() + (g0 + n) * s.batch_bytes());
    return p;
}
// card 0 memory the way ggml-sycl's pool allocates it (sycl_ext_oneapi_virtual_mem: physical memory mapped into a
// reserved range) - its temporaries (mul_mat's quantised activations) live there
namespace sx = sycl::ext::oneapi::experimental;
struct Vmm { uintptr_t va = 0; size_t size = 0; std::unique_ptr<sx::physical_mem> pm; void* p = nullptr; };
Vmm* vmm_alloc(sycl::queue& q, size_t bytes) {
    Vmm* v = new Vmm;
    const size_t gran = sx::get_mem_granularity(q.get_device(), q.get_context(), sx::granularity_mode::recommended);
    v->size = (bytes + gran - 1) / gran * gran;
    v->va = sx::reserve_virtual_mem(v->size, q.get_context());
    v->pm = std::make_unique<sx::physical_mem>(q.get_device(), q.get_context(), v->size);
    v->p = v->pm->map(v->va, v->size, sx::address_access_mode::read_write);
    return v;
}
uint8_t* upload(sycl::queue& q, const Planar& p) {
    uint8_t* d = sycl::malloc_device<uint8_t>(p.h.size(), q);
    q.memcpy(d, p.h.data(), p.h.size()).wait();
    return d;
}

fastmv::Args mv(const void* w, const Planar& p, const int8_t* xq, const float* xd, float* y) {
    fastmv::Args a;
    a.w = w; a.K = p.K; a.N = p.N; a.G = p.G; a.g_div = 1; a.n_blocks_total = p.nb(); a.w_batch_bytes = p.batch_bytes();
    a.x_col_stride = p.K; a.x_batch_stride = p.K;
    a.y = y; a.y_col_stride = p.N * p.G; a.y_batch_stride = p.N; a.ncols = 1;
    a.xq = xq; a.xd = xd;
    return a;
}
void mvl(sycl::queue& q, const void* w, const Planar& p, const int8_t* xq, const float* xd, float* y) {
    if (!fastmv::launch(q, fastmv::WType::Q6K_R8, mv(w, p, xq, xd, y))) { std::fprintf(stderr, "fastmv shape not covered\n"); std::exit(1); }
}
void qx(sycl::queue& q, const float* x, int64_t G, int64_t K, int8_t* xq, float* xd) { fastmv::quant_x(q, x, K, K, 1, G, K, xq, xd); }

sycl::event tiny(sycl::queue& q, float* p, int n) {
    return q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { p[i] = p[i] * 0.5f + 1.0f; });
}
sycl::event sum4(sycl::queue& q, float* y, const float* s, int64_t n, int parts, int64_t stride, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            float v = y[i];
            for (int k = 0; k < parts; ++k) v += s[k * stride + i];
            y[i] = v;
        });
    });
}
void swiglu(sycl::queue& q, const float* gu, int64_t F, float* h) {   // gu = [gate F | up F]
    q.parallel_for(sycl::range<1>((size_t) F), [=](sycl::id<1> i) {
        const float g = gu[i], u = gu[F + i];
        h[i] = g / (1.0f + sycl::exp(-g)) * u;
    });
}

sycl::event busy(Ctx& c, int64_t iters, int tok = 0) {
    float* p = c.busy_buf;
    volatile int* hf = c.hflag;
    return c.q[0].single_task([=] {
        float x = p[0];
        for (int64_t i = 0; i < iters; ++i) x = x * 0.9999999f + 1e-7f;
        p[1] = x;
        if (tok) hf[0] = tok;
    });
}
void close_gate(Ctx& c) { c.busy_ev = busy(c, c.busy_iters, c.tok + 1); }
bool gate_was_open(Ctx& c) {   // the busy kernel ended before the host finished queueing: the timing is host-bound
    return c.busy_ev.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

// time `iter` N times: gated (GPU time) and host-paced (wall); returns {gpu us/iter, host submit us/iter, paced us/iter}
struct T3 { double gpu, submit, paced; };
T3 timeit(Ctx& c, int N, const std::function<void()>& iter) {
    for (int i = 0; i < 10; ++i) iter();
    for (auto& q : c.q) q.wait();
    close_gate(c);
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i) iter();
    const double sub = us_since(t0);
    if (gate_was_open(c)) std::printf("  (warning: the gate opened before %d iterations were queued: %.0f us)\n", N, sub);
    // card 0's last command writes a host flag; the busy kernel wrote one at its end - the host spins on both (the
    // GPU's writes into pinned memory are seen within ~1 us, as the tier's timeline stamps)
    const int tok = ++c.tok;
    volatile int* hf = c.hflag;
    c.q[0].single_task([=] { hf[1] = tok; });
    while (hf[0] != tok) {}
    const auto t1 = clk::now();
    while (hf[1] != tok) {}
    const double gpu = us_since(t1);
    for (auto& q : c.q) q.wait();
    for (auto& q : c.q) q.wait();
    auto t2 = clk::now();
    for (int i = 0; i < N; ++i) iter();
    for (auto& q : c.q) q.wait();
    const double paced = us_since(t2);
    return {gpu / N, sub / N, paced / N};
}

double maxdiff(sycl::queue& q, const float* a, const float* b, int64_t n, double* amax) {
    std::vector<float> x((size_t) n), y((size_t) n);
    q.memcpy(x.data(), a, n * 4).wait();
    q.memcpy(y.data(), b, n * 4).wait();
    double md = 0, mx = 0;
    for (int64_t i = 0; i < n; ++i) { md = std::max(md, (double) std::fabs(x[i] - y[i])); mx = std::max(mx, (double) std::fabs(x[i])); }
    *amax = mx;
    return md;
}
}  // namespace

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 200;
    const std::string only = argc > 2 ? argv[2] : "";
    Ctx c;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        c.dev = p.get_devices(sycl::info::device_type::gpu);
        if (!c.dev.empty()) break;
    }
    if (c.dev.size() < 4) { std::printf("need 4 Level Zero GPUs\n"); return 1; }
    c.dev.resize(4);
    // platform default context (as ggml-sycl / the tier). No profiling: on a profiled queue every submission that
    // depends on another card's event took ~260 us (40 -> 794 us an iteration)
    for (auto& d : c.dev) c.q.emplace_back(d, sycl::property::queue::in_order());
    std::printf("contexts equal: %d\n", (int) (c.q[0].get_context() == c.q[1].get_context()));
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    c.hflag = sycl::malloc_host<int>(2, c.q[0]);
    c.hflag[0] = c.hflag[1] = 0;
    c.busy_buf = sycl::malloc_device<float>(2, c.q[0]);
    c.q[0].memset(c.busy_buf, 0, 8).wait();
    {   // calibrate the busy kernel to ~150 ms
        busy(c, 1 << 20).wait();
        auto t = clk::now();
        busy(c, 1 << 22).wait();
        const double us = us_since(t);
        c.busy_iters = (int64_t) ((double) (1 << 22) * 150000.0 / std::max(us, 1.0));
        std::printf("busy kernel: %.0f us for 2^22 iterations -> %lld iterations a gate\n", us, (long long) c.busy_iters);
    }
    sycl::queue& q0 = c.q[0];

    // card 0's slots for the other cards' partials (as the tier's dslots: SYCL USM on card 0)
    const int64_t H = 5120;
    float* slots = sycl::malloc_device<float>(3 * H, q0);
    q0.memset(slots, 0, 3 * H * 4).wait();

    if (only.empty() || only == "rt") {   // ---- bare round trip
        float* a0 = sycl::malloc_device<float>(8192, q0);
        float* y0 = sycl::malloc_device<float>(H, q0);
        q0.memset(a0, 0, 8192 * 4).wait(); q0.memset(y0, 0, H * 4).wait();
        std::vector<float*> ak(4), yk(4);
        for (int k = 1; k < 4; ++k) {
            ak[k] = sycl::malloc_device<float>(8192, c.q[k]); yk[k] = sycl::malloc_device<float>(H, c.q[k]);
            c.q[k].memset(ak[k], 0, 8192 * 4).wait(); c.q[k].memset(yk[k], 0, H * 4).wait();
        }
        for (int bytes : {64, 8192, 32768}) {
            auto iter = [&]() {
                const sycl::event e = tiny(q0, a0, 64);
                std::vector<sycl::event> ev;
                for (int k = 1; k < 4; ++k) {
                    c.q[k].submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(ak[k], a0, bytes); });
                    tiny(c.q[k], yk[k], 64);
                    ev.push_back(c.q[k].memcpy(slots + (k - 1) * H, yk[k], H * 4));
                }
                sum4(q0, y0, slots, 64, 3, H, ev);
            };
            const T3 t = timeit(c, N, iter);
            const T3 t1 = timeit(c, N, [&]() { tiny(q0, a0, 64); sum4(q0, y0, slots, 64, 3, H, {}); });
            std::printf("rt %5d B scatter + 20 KB reduce: gpu %6.1f us/iter (card 0 alone, same 2 kernels: %5.1f) | host submit %5.1f | host-paced %6.1f\n",
                        bytes, t.gpu, t1.gpu, t.submit, t.paced);
        }
        // one hop only: card k waits on card 0's event (no copy) and back
        {
            auto iter = [&]() {
                const sycl::event e = tiny(q0, a0, 64);
                std::vector<sycl::event> ev;
                for (int k = 1; k < 4; ++k) {
                    ev.push_back(c.q[k].submit([&](sycl::handler& h) {
                        h.depends_on(e);
                        float* p = yk[k];
                        h.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { p[i] += 1.0f; });
                    }));
                }
                sum4(q0, y0, slots, 64, 3, H, ev);
            };
            const T3 t = timeit(c, N, iter);
            std::printf("rt dependency only (no copies): gpu %6.1f us/iter | host submit %5.1f | host-paced %6.1f\n", t.gpu, t.submit, t.paced);
        }
    }

    if (only.empty() || only == "oproj") {   // ---- attention output: wo_a (grouped) + wo_b
        const Planar woa = make_planar(1024, 4096, 8, 11), wob = make_planar(H, 8192, 1, 12);
        uint8_t* woa0 = upload(q0, woa);
        uint8_t* wob0 = upload(q0, wob);
        std::vector<float> oh(8 * 4096);
        std::mt19937 rng(5);
        for (auto& v : oh) v = (float) ((int) (rng() % 200) - 100) / 100.0f;
        float* o0 = sycl::malloc_device<float>(8 * 4096, q0);
        q0.memcpy(o0, oh.data(), oh.size() * 4).wait();
        int8_t* xq0 = sycl::malloc_device<int8_t>(8 * 4096, q0);
        float* xd0 = sycl::malloc_device<float>(8 * 4096 / 32, q0);
        float* oa0 = sycl::malloc_device<float>(8192, q0);
        int8_t* xq0b = sycl::malloc_device<int8_t>(8192, q0);
        float* xd0b = sycl::malloc_device<float>(8192 / 32, q0);
        float* y1 = sycl::malloc_device<float>(H, q0);   // card 0 alone
        float* y4 = sycl::malloc_device<float>(H, q0);   // split
        auto one = [&]() {
            qx(q0, o0, 8, 4096, xq0, xd0);
            mvl(q0, woa0, woa, xq0, xd0, oa0);
            qx(q0, oa0, 1, 8192, xq0b, xd0b);
            mvl(q0, wob0, wob, xq0b, xd0b, y1);
        };
        // the split: card k gets groups 2k, 2k+1 of wo_a and wo_b's columns 2048 k .. 2048 k + 2047
        std::vector<Planar> pa(4), pb(4);
        std::vector<uint8_t*> wa(4), wb(4);
        std::vector<int8_t*> xqa(4), xqb(4);
        std::vector<float*> xda(4), xdb(4), oak(4), yk(4);
        for (int k = 0; k < 4; ++k) {
            pa[k] = batches_of(woa, 2 * k, 2);
            pb[k] = cols_of(wob, 2048 * k, 2048);
            wa[k] = upload(c.q[k], pa[k]);
            wb[k] = upload(c.q[k], pb[k]);
            xqa[k] = sycl::malloc_device<int8_t>(2 * 4096, c.q[k]);
            xda[k] = sycl::malloc_device<float>(2 * 4096 / 32, c.q[k]);
            oak[k] = sycl::malloc_device<float>(2048, c.q[k]);
            xqb[k] = sycl::malloc_device<int8_t>(2048, c.q[k]);
            xdb[k] = sycl::malloc_device<float>(2048 / 32, c.q[k]);
            yk[k] = k == 0 ? y4 : sycl::malloc_device<float>(H, c.q[k]);
        }
        auto split = [&]() {
            const sycl::event e = fastmv::quant_x(q0, o0, 4096, 4096, 1, 8, 4096, xq0, xd0);   // all 8 groups
            std::vector<sycl::event> ev;
            for (int k = 1; k < 4; ++k) {
                sycl::queue& q = c.q[k];
                q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(xqa[k], xq0 + 2 * k * 4096, 2 * 4096); });
                q.memcpy(xda[k], xd0 + 2 * k * 128, 2 * 128 * 4);
                mvl(q, wa[k], pa[k], xqa[k], xda[k], oak[k]);
                qx(q, oak[k], 1, 2048, xqb[k], xdb[k]);
                mvl(q, wb[k], pb[k], xqb[k], xdb[k], yk[k]);
                ev.push_back(q.memcpy(slots + (k - 1) * H, yk[k], H * 4));
            }
            mvl(q0, wa[0], pa[0], xq0, xd0, oak[0]);   // groups 0, 1 straight from card 0's xq
            qx(q0, oak[0], 1, 2048, xqb[0], xdb[0]);
            mvl(q0, wb[0], pb[0], xqb[0], xdb[0], y4);
            sum4(q0, y4, slots, H, 3, H, ev);
        };
        const T3 t1 = timeit(c, N, one);
        const T3 t4 = timeit(c, N, split);
        double amax = 0;
        const double md = maxdiff(q0, y1, y4, H, &amax);
        std::printf("oproj card 0 alone: gpu %6.1f us | host submit %5.1f | host-paced %6.1f\n", t1.gpu, t1.submit, t1.paced);
        std::printf("oproj four cards:   gpu %6.1f us | host submit %5.1f | host-paced %6.1f   max|diff| %.2e (max|y| %.2e)\n",
                    t4.gpu, t4.submit, t4.paced, md, amax);
        // card 0's quarter alone (the floor of the split)
        const T3 tq = timeit(c, N, [&]() {
            fastmv::quant_x(q0, o0, 4096, 4096, 1, 8, 4096, xq0, xd0);
            mvl(q0, wa[0], pa[0], xq0, xd0, oak[0]); qx(q0, oak[0], 1, 2048, xqb[0], xdb[0]); mvl(q0, wb[0], pb[0], xqb[0], xdb[0], y4);
        });
        std::printf("oproj card 0's quarter only: gpu %6.1f us\n", tq.gpu);
        {   // the same split synchronised the way the tier's mv_hook does it (barriers instead of event dependencies)
            auto split_b = [&]() {
                fastmv::quant_x(q0, o0, 4096, 4096, 1, 8, 4096, xq0, xd0);
                const sycl::event e = q0.ext_oneapi_submit_barrier();
                std::vector<sycl::event> ev;
                for (int k = 1; k < 4; ++k) {
                    sycl::queue& q = c.q[k];
                    q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(xqa[k], xq0 + 2 * k * 4096, 2 * 4096); });
                    q.memcpy(xda[k], xd0 + 2 * k * 128, 2 * 128 * 4);
                    mvl(q, wa[k], pa[k], xqa[k], xda[k], oak[k]);
                    qx(q, oak[k], 1, 2048, xqb[k], xdb[k]);
                    mvl(q, wb[k], pb[k], xqb[k], xdb[k], yk[k]);
                    q.memcpy(slots + (k - 1) * H, yk[k], H * 4);
                    ev.push_back(q.ext_oneapi_submit_barrier());
                }
                mvl(q0, wa[0], pa[0], xq0, xd0, oak[0]);
                qx(q0, oak[0], 1, 2048, xqb[0], xdb[0]);
                mvl(q0, wb[0], pb[0], xqb[0], xdb[0], y4);
                q0.ext_oneapi_submit_barrier(ev);
                sum4(q0, y4, slots, H, 3, H, {});
            };
            const T3 tb = timeit(c, N, split_b);
            const double mdb = maxdiff(q0, y1, y4, H, &amax);
            std::printf("oproj four cards, barrier sync (as mv_hook): gpu %6.1f us | host submit %5.1f | host-paced %6.1f   max|diff| %.2e\n",
                        tb.gpu, tb.submit, tb.paced, mdb);
        }
        {   // the same split, card 0's xq / xd in ggml-sycl-style VMM memory
            Vmm* vq = vmm_alloc(q0, 8 * 4096 + 8 * 4096 / 32 * 4 + 4096);
            int8_t* vxq = (int8_t*) vq->p;
            float* vxd = (float*) ((uint8_t*) vq->p + 8 * 4096);
            auto split_v = [&]() {
                const sycl::event e = fastmv::quant_x(q0, o0, 4096, 4096, 1, 8, 4096, vxq, vxd);
                std::vector<sycl::event> ev;
                for (int k = 1; k < 4; ++k) {
                    sycl::queue& q = c.q[k];
                    q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(xqa[k], vxq + 2 * k * 4096, 2 * 4096); });
                    q.memcpy(xda[k], vxd + 2 * k * 128, 2 * 128 * 4);
                    mvl(q, wa[k], pa[k], xqa[k], xda[k], oak[k]);
                    qx(q, oak[k], 1, 2048, xqb[k], xdb[k]);
                    mvl(q, wb[k], pb[k], xqb[k], xdb[k], yk[k]);
                    ev.push_back(q.memcpy(slots + (k - 1) * H, yk[k], H * 4));
                }
                mvl(q0, wa[0], pa[0], vxq, vxd, oak[0]);
                qx(q0, oak[0], 1, 2048, xqb[0], xdb[0]);
                mvl(q0, wb[0], pb[0], xqb[0], xdb[0], y4);
                sum4(q0, y4, slots, H, 3, H, ev);
            };
            const T3 tv = timeit(c, N, split_v);
            const double mdv = maxdiff(q0, y1, y4, H, &amax);
            std::printf("oproj four cards, x from VMM memory: gpu %6.1f us | host submit %5.1f | host-paced %6.1f   max|diff| %.2e\n",
                        tv.gpu, tv.submit, tv.paced, mdv);
        }
    }

    if (only.empty() || only == "qb") {   // ---- q_b, row split (heads stay on their card)
        const Planar wq = make_planar(32768, 1280, 1, 13);
        uint8_t* w0 = upload(q0, wq);
        std::vector<float> xh(1280);
        std::mt19937 rng(6);
        for (auto& v : xh) v = (float) ((int) (rng() % 200) - 100) / 100.0f;
        float* x0 = sycl::malloc_device<float>(1280, q0);
        q0.memcpy(x0, xh.data(), 1280 * 4).wait();
        int8_t* xq0 = sycl::malloc_device<int8_t>(1280, q0);
        float* xd0 = sycl::malloc_device<float>(40, q0);
        float* y0 = sycl::malloc_device<float>(32768, q0);
        std::vector<Planar> pk(4);
        std::vector<uint8_t*> wk(4);
        std::vector<int8_t*> xqk(4);
        std::vector<float*> xdk(4), yk(4);
        for (int k = 0; k < 4; ++k) {
            pk[k] = rows_of(wq, 8192 * k, 8192);
            wk[k] = upload(c.q[k], pk[k]);
            xqk[k] = sycl::malloc_device<int8_t>(1280, c.q[k]);
            xdk[k] = sycl::malloc_device<float>(40, c.q[k]);
            yk[k] = sycl::malloc_device<float>(8192, c.q[k]);
        }
        float* done = sycl::malloc_device<float>(64, q0);
        q0.memset(done, 0, 256).wait();
        const T3 t1 = timeit(c, N, [&]() { qx(q0, x0, 1, 1280, xq0, xd0); mvl(q0, w0, wq, xq0, xd0, y0); });
        const T3 t4 = timeit(c, N, [&]() {
            const sycl::event e = fastmv::quant_x(q0, x0, 1280, 1280, 1, 1, 1280, xq0, xd0);
            std::vector<sycl::event> ev;
            for (int k = 1; k < 4; ++k) {
                c.q[k].submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(xqk[k], xq0, 1280); });
                c.q[k].memcpy(xdk[k], xd0, 160);
                mvl(c.q[k], wk[k], pk[k], xqk[k], xdk[k], yk[k]);
                ev.push_back(c.q[k].memcpy(slots + (k - 1) * H, yk[k], 64 * 4));   // a token "done" back (the heads stay)
            }
            mvl(q0, wk[0], pk[0], xq0, xd0, yk[0]);
            sum4(q0, done, slots, 64, 3, H, ev);
        });
        std::printf("q_b card 0 alone: gpu %6.1f us | host submit %5.1f | host-paced %6.1f\n", t1.gpu, t1.submit, t1.paced);
        std::printf("q_b four cards:   gpu %6.1f us | host submit %5.1f | host-paced %6.1f\n", t4.gpu, t4.submit, t4.paced);
    }

    if (only.empty() || only == "shexp") {   // ---- shared expert: card 0 vs cards 1-3 (768 of the 2304 each)
        const int64_t F = 2304;
        const Planar gu = make_planar(2 * F, H, 1, 14), dn = make_planar(H, F, 1, 15);
        uint8_t* gu0 = upload(q0, gu);
        uint8_t* dn0 = upload(q0, dn);
        std::vector<float> xh((size_t) H);
        std::mt19937 rng(8);
        for (auto& v : xh) v = (float) ((int) (rng() % 200) - 100) / 1000.0f;
        float* x0 = sycl::malloc_device<float>(H, q0);
        q0.memcpy(x0, xh.data(), H * 4).wait();
        int8_t* xq0 = sycl::malloc_device<int8_t>(H, q0);
        float* xd0 = sycl::malloc_device<float>(H / 32, q0);
        float* g0 = sycl::malloc_device<float>(2 * F, q0);
        float* h0 = sycl::malloc_device<float>(F, q0);
        int8_t* hq0 = sycl::malloc_device<int8_t>(F, q0);
        float* hd0 = sycl::malloc_device<float>(F / 32, q0);
        float* y1 = sycl::malloc_device<float>(H, q0);
        float* y4 = sycl::malloc_device<float>(H, q0);
        const T3 t1 = timeit(c, N, [&]() {
            qx(q0, x0, 1, H, xq0, xd0); mvl(q0, gu0, gu, xq0, xd0, g0); swiglu(q0, g0, F, h0); qx(q0, h0, 1, F, hq0, hd0); mvl(q0, dn0, dn, hq0, hd0, y1);
        });
        // card k (1..3): gate rows [768 (k-1), +768) and the matching up rows; down's columns [768 (k-1), +768)
        const int64_t S = 768;
        std::vector<Planar> pg(4), pd(4);
        std::vector<uint8_t*> wg(4), wd(4);
        std::vector<int8_t*> xqk(4), hqk(4);
        std::vector<float*> xdk(4), gk(4), hk(4), hdk(4), yk(4);
        for (int k = 1; k < 4; ++k) {
            const Planar ga = rows_of(gu, S * (k - 1), S), ub = rows_of(gu, F + S * (k - 1), S);
            pg[k].N = 2 * S; pg[k].K = H; pg[k].G = 1;
            pg[k].h.resize(pg[k].batch_bytes());
            {   // [gate S | up S] in one planar matrix: per plane, gate's then up's blocks
                const int64_t nb1 = ga.nb(), nb2 = pg[k].nb();
                const size_t po[4] = {0, 128, 192, 208}, pb2[4] = {128, 64, 16, 2};
                for (int p = 0; p < 4; ++p) {
                    std::memcpy(pg[k].h.data() + po[p] * nb2, ga.h.data() + po[p] * nb1, pb2[p] * nb1);
                    std::memcpy(pg[k].h.data() + po[p] * nb2 + pb2[p] * nb1, ub.h.data() + po[p] * nb1, pb2[p] * nb1);
                }
            }
            pd[k] = cols_of(dn, S * (k - 1), S);
            wg[k] = upload(c.q[k], pg[k]);
            wd[k] = upload(c.q[k], pd[k]);
            xqk[k] = sycl::malloc_device<int8_t>(H, c.q[k]);
            xdk[k] = sycl::malloc_device<float>(H / 32, c.q[k]);
            gk[k] = sycl::malloc_device<float>(2 * S, c.q[k]);
            hk[k] = sycl::malloc_device<float>(S, c.q[k]);
            hqk[k] = sycl::malloc_device<int8_t>(S, c.q[k]);
            hdk[k] = sycl::malloc_device<float>(S / 32, c.q[k]);
            yk[k] = sycl::malloc_device<float>(H, c.q[k]);
        }
        float* zero = sycl::malloc_device<float>(H, q0);
        q0.memset(zero, 0, H * 4).wait();
        const T3 t3 = timeit(c, N, [&]() {
            const sycl::event e = fastmv::quant_x(q0, x0, H, H, 1, 1, H, xq0, xd0);
            std::vector<sycl::event> ev;
            for (int k = 1; k < 4; ++k) {
                sycl::queue& q = c.q[k];
                q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(xqk[k], xq0, H); });
                q.memcpy(xdk[k], xd0, H / 32 * 4);
                mvl(q, wg[k], pg[k], xqk[k], xdk[k], gk[k]);
                swiglu(q, gk[k], S, hk[k]);
                qx(q, hk[k], 1, S, hqk[k], hdk[k]);
                mvl(q, wd[k], pd[k], hqk[k], hdk[k], yk[k]);
                ev.push_back(q.memcpy(slots + (k - 1) * H, yk[k], H * 4));
            }
            q0.memcpy(y4, zero, H * 4);
            sum4(q0, y4, slots, H, 3, H, ev);
        });
        double amax = 0;
        const double md = maxdiff(q0, y1, y4, H, &amax);
        std::printf("shexp card 0 alone: gpu %6.1f us | host submit %5.1f | host-paced %6.1f\n", t1.gpu, t1.submit, t1.paced);
        std::printf("shexp cards 1-3:    gpu %6.1f us | host submit %5.1f | host-paced %6.1f   max|diff| %.2e (max|y| %.2e)\n",
                    t3.gpu, t3.submit, t3.paced, md, amax);
    }
    for (auto& q : c.q) q.wait();
    return 0;
}
