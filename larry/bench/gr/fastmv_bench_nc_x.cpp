// fastmv_bench.cpp - the fastmv kernels at DeepSeek-V4.1's dense decode shapes: correctness vs a CPU reference
// (standard Q6_K layout, reordered for the device as ggml-sycl does) and speed.
#include "fastmv.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct BQ6 { uint8_t ql[128]; uint8_t qh[64]; int8_t sc[16]; uint16_t d; };
static_assert(sizeof(BQ6) == 210, "q6_K");

static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>((sycl::half) f); }

static void deq_q6(const BQ6 & b, float * y) {
    const float d = h2f(b.d);
    const uint8_t * ql = b.ql; const uint8_t * qh = b.qh; const int8_t * sc = b.sc;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            y[l + 0] = d * sc[is + 0] * (int8_t) (((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * (int8_t) (((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> ub(0, 255);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    struct S { const char * name; int64_t K, N, G; fastmv::WType t; };
    const S shapes[] = {
        {"q_b 1280x32768", 1280, 32768, 1, fastmv::WType::Q6K_R}, {"o_b 8192x5120", 8192, 5120, 1, fastmv::WType::Q6K_R},
        {"o_a 4096x1024 x8", 4096, 1024, 8, fastmv::WType::Q6K_R}, {"o_a STD 4096x1024 x8", 4096, 1024, 8, fastmv::WType::Q6K},
        {"q_b STD 1280x32768", 1280, 32768, 1, fastmv::WType::Q6K}, {"shexp gu 5120x2304", 5120, 2304, 1, fastmv::WType::Q6K_R},
        {"shexp d 2304x5120", 2304, 5120, 1, fastmv::WType::Q6K_R}, {"q_a 5120x1280", 5120, 1280, 1, fastmv::WType::Q6K_R},
        {"kv 5120x512", 5120, 512, 1, fastmv::WType::Q6K_R}, {"hc F16 20480x24", 20480, 24, 1, fastmv::WType::F16},
        {"q_b R8 1280x32768", 1280, 32768, 1, fastmv::WType::Q6K_R8}, {"o_b R8 8192x5120", 8192, 5120, 1, fastmv::WType::Q6K_R8},
        {"shexp gu R8 5120x2304", 5120, 2304, 1, fastmv::WType::Q6K_R8}, {"shexp d R8 2304x5120", 2304, 5120, 1, fastmv::WType::Q6K_R8},
        {"q_a R8 5120x1280", 5120, 1280, 1, fastmv::WType::Q6K_R8}, {"kv R8 5120x512", 5120, 512, 1, fastmv::WType::Q6K_R8},
        {"qkv R8 5120x1792", 5120, 1792, 1, fastmv::WType::Q6K_R8}, {"router F16 5120x384", 5120, 384, 1, fastmv::WType::F16},
        {"router F32 5120x384", 5120, 384, 1, fastmv::WType::F32}, {"head 5120x129280", 5120, 129280, 1, fastmv::WType::Q6K_R}};
    for (const S & s : shapes) {
        const int64_t rows = s.N * s.G;
        size_t wbytes;
        std::vector<uint8_t> wdev;
        std::vector<float> wf;   // dequantised reference weights (only for checking a few rows)
        const int64_t bpr = s.K / 256;
        const bool q6 = s.t == fastmv::WType::Q6K_R || s.t == fastmv::WType::Q6K || s.t == fastmv::WType::Q6K_R8;
        if (q6) {
            const int64_t nb = rows * bpr;
            std::vector<BQ6> blk((size_t) nb);
            for (auto & b : blk) {
                for (auto & v : b.ql) v = (uint8_t) ub(rng);
                for (auto & v : b.qh) v = (uint8_t) ub(rng);
                for (auto & v : b.sc) v = (int8_t) (ub(rng) % 64 - 32);
                b.d = f2h(0.001f + 0.002f * (ub(rng) / 255.f));
            }
            wbytes = (size_t) nb * 210;
            wdev.resize(wbytes);
            const int64_t pb = s.N * bpr;   // blocks per batch plane set
            if (s.t == fastmv::WType::Q6K) std::memcpy(wdev.data(), blk.data(), wbytes);   // R8: reordered like Q6K_R
            else
            for (int64_t i = 0; i < nb; ++i) {   // ggml-sycl's reorder (per batch): ql | qh | scales | d planes
                const int64_t g = i / pb, j = i % pb;
                uint8_t * P = &wdev[(size_t) g * pb * 210];
                std::memcpy(P + (size_t) j * 128, blk[(size_t) i].ql, 128);
                std::memcpy(P + (size_t) pb * 128 + (size_t) j * 64, blk[(size_t) i].qh, 64);
                std::memcpy(P + (size_t) pb * 192 + (size_t) j * 16, blk[(size_t) i].sc, 16);
                std::memcpy(P + (size_t) pb * 208 + (size_t) j * 2, &blk[(size_t) i].d, 2);
            }
            // reference rows: 0, mid, last
            wf.resize((size_t) 3 * s.K);
            const int64_t rr[3] = {0, rows / 2, rows - 1};
            for (int k = 0; k < 3; ++k)
                for (int64_t b = 0; b < bpr; ++b) deq_q6(blk[(size_t) (rr[k] * bpr + b)], &wf[(size_t) k * s.K + b * 256]);
        } else {
            const size_t es = s.t == fastmv::WType::F16 ? 2 : 4;
            wbytes = (size_t) rows * s.K * es;
            wdev.resize(wbytes);
            wf.resize((size_t) 3 * s.K);
            const int64_t rr[3] = {0, rows / 2, rows - 1};
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t k = 0; k < s.K; ++k) {
                    const float v = uf(rng) * 0.05f;
                    if (es == 2) { const uint16_t h = f2h(v); std::memcpy(&wdev[(size_t) (r * s.K + k) * 2], &h, 2); }
                    else std::memcpy(&wdev[(size_t) (r * s.K + k) * 4], &v, 4);
                    for (int j = 0; j < 3; ++j) if (rr[j] == r) wf[(size_t) j * s.K + k] = es == 2 ? h2f(f2h(v)) : v;
                }
        }
        for (int nc : {1, 2, 3, 4, 5, 6, 8}) {
            std::vector<float> x((size_t) s.G * nc * s.K);
            for (auto & v : x) v = uf(rng);
            uint8_t * dw = sycl::malloc_device<uint8_t>(wbytes, q);
            float * dx = sycl::malloc_device<float>(x.size(), q);
            float * dy = sycl::malloc_device<float>((size_t) s.G * nc * s.N, q);
            q.memcpy(dw, wdev.data(), wbytes);
            q.memcpy(dx, x.data(), x.size() * 4).wait();
            fastmv::Args a;
            a.w = dw; a.K = s.K; a.N = s.N; a.G = s.G; a.g_div = 1; a.n_blocks_total = s.N * bpr;
            a.w_row_bytes = s.t == fastmv::WType::Q6K ? (size_t) bpr * 210 : (size_t) s.K * (s.t == fastmv::WType::F16 ? 2 : 4);
            a.w_batch_bytes = q6 ? (size_t) s.N * bpr * 210 : a.w_row_bytes * s.N;
            int8_t * dxq = sycl::malloc_device<int8_t>((size_t) s.G * nc * s.K, q);
            float * dxd = sycl::malloc_device<float>((size_t) s.G * nc * s.K / 32, q);
            a.xq = dxq; a.xd = dxd;
            auto launch_all = [&] {
                if (s.t == fastmv::WType::Q6K_R8) fastmv::quant_x(q, dx, s.K, (int64_t) nc * s.K, nc, s.G, s.K, dxq, dxd);
                fastmv::launch(q, s.t, a);
            };
            a.x = dx; a.x_col_stride = s.K; a.x_batch_stride = (int64_t) nc * s.K;
            a.y = dy; a.y_col_stride = s.N; a.y_batch_stride = (int64_t) nc * s.N; a.ncols = nc;
            launch_all();
            q.wait();
            std::vector<float> y((size_t) s.G * nc * s.N);
            q.memcpy(y.data(), dy, y.size() * 4).wait();
            // check: rows 0 / mid / last for every column
            double worst = 0;
            const int64_t rr[3] = {0, rows / 2, rows - 1};
            for (int j = 0; j < 3; ++j) {
                const int64_t g = rr[j] / s.N, r = rr[j] % s.N;
                for (int c = 0; c < nc; ++c) {
                    double ref = 0, mag = 0;
                    for (int64_t k = 0; k < s.K; ++k) {
                        const double p = (double) wf[(size_t) j * s.K + k] * x[(size_t) (g * nc + c) * s.K + k];
                        ref += p; mag += std::fabs(p);
                    }
                    const double got = y[(size_t) (g * nc + c) * s.N + r];
                    worst = std::max(worst, std::fabs(got - ref) / (mag + 1e-30));
                }
            }
            for (int w = 0; w < 5; ++w) launch_all();
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) launch_all();
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::printf("%-22s ncols %d: %8.1f us  %6.0f GB/s  err %.1e %s\n", s.name, nc, us, wbytes / us / 1e3, worst,
                        worst < (s.t == fastmv::WType::Q6K_R8 ? 3e-2 : 1e-5) ? "ok" : "FAIL");
            sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q); sycl::free(dxq, q); sycl::free(dxd, q);
        }
    }
    return 0;
}
