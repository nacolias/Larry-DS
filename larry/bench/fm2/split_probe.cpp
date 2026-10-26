// split_probe.cpp - one Q6K_R8 mat-vec (DeepSeek-V4.1's o_b / q_b) on card 0 alone vs split by rows over four cards:
// card 0 computes its quarter; cards 1-3 get the quantised x (a copy after card 0's barrier), run their quarter from a
// local copy of their rows' planes and write y straight into card 0's buffer (peer writes); card 0 waits on them.
// Prints us per call (back to back, each call ordered after the previous) and the max |difference| of the two ys.
#include "fastmv.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 300;
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 4) { std::printf("need 4 GPUs\n"); return 1; }
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto & d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    for (int k = 1; k < 4; ++k) {
        if (!gpus[k].ext_oneapi_can_access_peer(gpus[0])) { std::printf("card %d cannot access card 0\n", k); return 1; }
        gpus[k].ext_oneapi_enable_peer_access(gpus[0]);
    }
    struct S { const char * name; int64_t K, N; };
    for (const S s : {S{"o_b 8192x5120", 8192, 5120}, S{"q_b 1280x32768", 1280, 32768}}) {
        const int64_t bpr = s.K / 256, nb = s.N * bpr;
        // random R8 planes (ql 128 | qh 64 | sc 16 | d 2 per block), small d so sums stay finite
        std::mt19937 rng(7);
        std::vector<uint8_t> w((size_t) nb * 210);
        for (size_t i = 0; i < (size_t) nb * 208; ++i) w[i] = (uint8_t) (rng() & 0xFF);
        for (int64_t b = 0; b < nb; ++b) { const sycl::half h = (sycl::half) 0.001f; std::memcpy(&w[208 * nb + 2 * b], &h, 2); }
        for (size_t i = 192 * nb; i < (size_t) 208 * nb; ++i) w[i] = (uint8_t) (rng() % 64);   // scales >= 0, small
        uint8_t * w0 = sycl::malloc_device<uint8_t>(w.size(), q[0]);
        q[0].memcpy(w0, w.data(), w.size()).wait();
        std::vector<float> xh((size_t) s.K);
        for (auto & v : xh) v = (float) ((int) (rng() % 200) - 100) / 100.0f;
        float * x0 = sycl::malloc_device<float>(s.K, q[0]);
        int8_t * xq0 = sycl::malloc_device<int8_t>(s.K, q[0]);
        float * xd0 = sycl::malloc_device<float>(s.K / 32, q[0]);
        float * y0 = sycl::malloc_device<float>(s.N, q[0]);
        float * y1 = sycl::malloc_device<float>(s.N, q[0]);
        q[0].memcpy(x0, xh.data(), s.K * 4).wait();
        fastmv::quant_x(q[0], x0, s.K, s.K, 1, 1, s.K, xq0, xd0).wait();
        auto args = [&](const void * wb, int64_t nbt, int64_t N, const int8_t * xq, const float * xd, float * y) {
            fastmv::Args a;
            a.w = wb; a.K = s.K; a.N = N; a.n_blocks_total = nbt; a.x = x0; a.x_col_stride = s.K; a.x_batch_stride = s.K;
            a.y = y; a.y_col_stride = s.N; a.y_batch_stride = s.N; a.ncols = 1; a.xq = xq; a.xd = xd;
            return a;
        };
        // one card
        for (int i = 0; i < 20; ++i) fastmv::launch(q[0], fastmv::WType::Q6K_R8, args(w0, nb, s.N, xq0, xd0, y0));
        q[0].wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fastmv::launch(q[0], fastmv::WType::Q6K_R8, args(w0, nb, s.N, xq0, xd0, y0));
        q[0].wait();
        const double us1 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        // the split: rows [r[k], r[k+1]) on card k
        const int64_t r[5] = {0, s.N / 4, s.N / 2, 3 * s.N / 4, s.N};
        std::vector<uint8_t *> wk(4, nullptr);
        std::vector<int8_t *> xqk(4, nullptr);
        std::vector<float *> xdk(4, nullptr);
        for (int k = 1; k < 4; ++k) {
            const int64_t b0 = r[k] * bpr, bn = (r[k + 1] - r[k]) * bpr;
            wk[k] = sycl::malloc_device<uint8_t>((size_t) bn * 210, q[k]);
            const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};   // plane offsets / bytes per block
            for (int p = 0; p < 4; ++p)
                q[k].memcpy(wk[k] + po[p] * bn, w0 + po[p] * nb + pb[p] * b0, pb[p] * bn).wait();
            xqk[k] = sycl::malloc_device<int8_t>(s.K, q[k]);
            xdk[k] = sycl::malloc_device<float>(s.K / 32, q[k]);
        }
        auto split_call = [&](float * y) {
            sycl::event e0 = q[0].ext_oneapi_submit_barrier();
            std::vector<sycl::event> ek;
            for (int k = 1; k < 4; ++k) {
                q[k].submit([&](sycl::handler & h) { h.depends_on(e0); h.memcpy(xqk[k], xq0, s.K); });
                q[k].memcpy(xdk[k], xd0, s.K / 32 * 4);
                fastmv::launch(q[k], fastmv::WType::Q6K_R8, args(wk[k], (r[k + 1] - r[k]) * bpr, r[k + 1] - r[k], xqk[k], xdk[k], y + r[k]));
                ek.push_back(q[k].ext_oneapi_submit_barrier());
            }
            fastmv::launch(q[0], fastmv::WType::Q6K_R8, args(w0, nb, r[1], xq0, xd0, y));
            q[0].ext_oneapi_submit_barrier(ek);
        };
        for (int i = 0; i < 20; ++i) split_call(y1);
        q[0].wait();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) split_call(y1);
        q[0].wait();
        const double us4 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::vector<float> a(s.N), b(s.N);
        q[0].memcpy(a.data(), y0, s.N * 4).wait();
        q[0].memcpy(b.data(), y1, s.N * 4).wait();
        double md = 0, mx = 0;
        for (int64_t i = 0; i < s.N; ++i) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mx = std::max(mx, (double) std::fabs(a[i])); }
        std::printf("%-16s one card %6.1f us   four cards %6.1f us   max|diff| %.2e (max|y| %.2e)\n", s.name, us1, us4, md, mx);
        for (int k = 1; k < 4; ++k) { sycl::free(wk[k], ctx); sycl::free(xqk[k], ctx); sycl::free(xdk[k], ctx); }
        for (void * p : {(void *) w0, (void *) x0, (void *) xq0, (void *) xd0, (void *) y0, (void *) y1}) sycl::free(p, ctx);
    }
    return 0;
}
