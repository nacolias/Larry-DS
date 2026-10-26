// xmx_mv.cpp - larry: prototype of a multi-column (DSpark verify: 2..8 tokens) mat-vec for ggml-sycl's planar Q6_K
// (ql | qh | scales | d planes) on the B70's XMX engines, against fastmv's dp4a r8m kernel and a CPU reference.
//   * a sub-group owns 16 weight rows; lane n decodes ITS row into the B fragment (lane n = column n of B: bench/dqb/
//     lpr_bw showed one-row-per-lane 16-byte loads stream at 480-560 GB/s), 16 values a k-step = one Q6_K scale group,
//     unpacked two at a time as fp16 (h(0x6400 | q) = 1024 + q) and scaled by d * sc in fp16;
//   * A = the columns' int8 activations (quant_x's rounding: one scale per 32 values), transposed to xT[k][8] as fp16
//     integers (lane k loads its 8 tokens in one 16-byte load); B = (q - 32) * sc, integers too: a 32-value block's
//     two joint_matrix_mads (8 x 16 x 16) are an exact integer dot product in f32, then scaled by xd[token] * d[row];
//   * K split over KS sub-groups of a work-group (reduced in local memory) when the matrix has few rows.
// build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" -I<ggml-sycl dir> xmx_mv.cpp -o xmx_mv
#include "fastmv.cpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using u4 = sycl::vec<uint32_t, 4>;
using h2 = sycl::vec<sycl::half, 2>;

struct BQ6 { uint8_t ql[128]; uint8_t qh[64]; int8_t sc[16]; uint16_t d; };
static_assert(sizeof(BQ6) == 210, "q6_K");
static float hf(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t fh(float f) { return sycl::bit_cast<uint16_t>((sycl::half) f); }
static void deq_q6(const BQ6 & b, float * y) {
    const float d = hf(b.d);
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

// int8 activations, transposed: xT[g][k][8] = round(x[g][t][k] / xdT[g][k / 32][t]) as fp16 (t < ncols; 0 above),
// xdT[g][j][8] = amax / 127 of x[g][t][32 j .. 32 j + 31] - quant_x's rounding in a layout the A fragment loads whole
static void quant_xT(sycl::queue & q, const float * x, int64_t xcs, int64_t xbs, int ncols, int64_t G, int64_t K,
                     sycl::half * xT, float * xdT) {
    const int64_t nj = K / 32;
    q.parallel_for(sycl::range<1>((size_t) (G * nj)), [=](sycl::id<1> id) {
        const int64_t g = (int64_t) id[0] / nj, j = (int64_t) id[0] % nj;
        float id_[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float m = 0.0f;
            if (t < ncols) {
                const float * p = x + g * xbs + t * xcs + 32 * j;
#pragma unroll
                for (int i = 0; i < 32; ++i) m = sycl::fmax(m, sycl::fabs(p[i]));
            }
            id_[t] = m > 0.0f ? 127.0f / m : 0.0f;
            xdT[(g * nj + j) * 8 + t] = m / 127.0f;
        }
        for (int i = 0; i < 32; ++i) {
            sycl::vec<sycl::half, 8> v;
#pragma unroll
            for (int t = 0; t < 8; ++t) v[t] = t < ncols ? (sycl::half) sycl::rint(x[g * xbs + t * xcs + 32 * j + i] * id_[t]) : (sycl::half) 0.0f;
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(xT + (g * K + 32 * j + i) * 8) = v;
        }
    });
}

// y[g][t][row] = sum_k W[g / g_div][row][k] x[g][t][k]; a sub-group owns R groups of 16 rows (sharing every A
// fragment); KS sub-groups a work-group split the rows' blocks
template <int KS, int R>
static sycl::event xmx_mv(sycl::queue & q, const void * w, int64_t N, int64_t K, int64_t G, int64_t g_div, int64_t nb_tot,
                          size_t w_batch_bytes, const sycl::half * xT, const float * xdT, int ncols, float * y,
                          int64_t ycs, int64_t ybs) {
    const int64_t bpr = K / 256, nrg = (N + 16 * R - 1) / (16 * R);
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(KS > 1 ? (size_t) KS * R * 16 * 8 : 1), h);
        auto kern = [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g = it.get_global_id(0);
            const int64_t rg = it.get_group(1);
            const sycl::sub_group sg = it.get_sub_group();
            const int ks = (int) sg.get_group_linear_id();
            const int ln = (int) sg.get_local_linear_id();
            int64_t row[R];
#pragma unroll
            for (int i = 0; i < R; ++i) row[i] = sycl::min((rg * R + i) * 16 + ln, N - 1);   // a tail lane decodes the last row
            const uint8_t * base = (const uint8_t *) w + (g / g_div) * w_batch_bytes;
            const uint8_t * ql_all = base;
            const uint8_t * qh_all = base + 128 * nb_tot;
            const int8_t * sc_all = (const int8_t *) (base + 192 * nb_tot);
            const uint16_t * d_all = (const uint16_t *) (base + 208 * nb_tot);
            const sycl::half * xg = xT + g * K * 8;
            const int64_t per = (bpr + KS - 1) / KS;
            const int64_t b0 = ks * per, b1 = sycl::min(bpr, b0 + per);
            const float * xdg = xdT + g * (K / 32) * 8;
            float c[R][8];
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int t = 0; t < 8; ++t) c[i][t] = 0.0f;
            const h2 off((sycl::half) 1056.0f, (sycl::half) 1056.0f);
            for (int64_t b = b0; b < b1; ++b) {
                u4 SC[R];
                float d[R];
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const int64_t blk = row[i] * bpr + b;
                    SC[i] = *reinterpret_cast<const u4 *>(sc_all + blk * 16);
                    d[i] = (float) sycl::bit_cast<sycl::half>(d_all[blk]);
                }
                const sycl::half * xb = xg + b * 256 * 8;
#pragma unroll
                for (int ipx = 0; ipx < 2; ++ipx) {   // half ipx: its ql (64 B) and qh (32 B) in registers
                    u4 QL[R][4], QH[R][2];
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int64_t blk = row[i] * bpr + b;
                        const u4 * qlp = reinterpret_cast<const u4 *>(ql_all + blk * 128) + 4 * ipx;
                        const u4 * qhp = reinterpret_cast<const u4 *>(qh_all + blk * 64) + 2 * ipx;
#pragma unroll
                        for (int u = 0; u < 4; ++u) QL[i][u] = qlp[u];
#pragma unroll
                        for (int u = 0; u < 2; ++u) QH[i][u] = qhp[u];
                    }
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {   // 32-value block j: scale groups 2 j, 2 j + 1, activation scale j
                        const int j = 4 * ipx + jj;
                        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> acc[R];
#pragma unroll
                        for (int i = 0; i < R; ++i) mx::joint_matrix_fill(sg, acc[i], 0.0f);
#pragma unroll
                        for (int mm = 0; mm < 2; ++mm) {   // values 16 m .. 16 m + 15: scale group m
                            const int m = 2 * j + mm;
                            const int k = (m % 8) / 2, hh = m % 2;
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> ma;
                            const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(xb + (16 * m + ln) * 8);
                            auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
                            for (int r = 0; r < 8; ++r) wa[r] = a8[r];
#pragma unroll
                            for (int i = 0; i < R; ++i) {
                                const u4 qlw = QL[i][2 * (k & 1) + hh];
                                const u4 qhw = QH[i][hh];
                                const int8_t scm = (int8_t) ((SC[i][m / 4] >> (8 * (m % 4))) & 0xFF);
                                const sycl::half sh = (sycl::half) (float) scm;
                                const h2 s2(sh, sh);
                                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> mb;
                                auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
                                for (int wd = 0; wd < 4; ++wd) {   // 4 values a word
                                    const uint32_t nib = (k & 2) ? ((qlw[wd] >> 4) & 0x0F0F0F0Fu) : (qlw[wd] & 0x0F0F0F0Fu);
                                    const uint32_t q6 = nib | (((qhw[wd] >> (2 * k)) & 0x03030303u) << 4);
                                    const h2 lo = (sycl::bit_cast<h2>((q6 & 0x00FF00FFu) | 0x64006400u) - off) * s2;          // v, v + 2
                                    const h2 hi = (sycl::bit_cast<h2>(((q6 >> 8) & 0x00FF00FFu) | 0x64006400u) - off) * s2;   // v + 1, v + 3
                                    wb[4 * wd + 0] = lo[0]; wb[4 * wd + 1] = hi[0]; wb[4 * wd + 2] = lo[1]; wb[4 * wd + 3] = hi[1];
                                }
                                mx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                            }
                        }
                        // the block's exact integer sums, times xd[token] * d[row]
                        const sycl::float4 * xdp = reinterpret_cast<const sycl::float4 *>(xdg + (b * 8 + j) * 8);
                        const sycl::float4 x0 = xdp[0], x1 = xdp[1];
                        const float xs[8] = {x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()};
#pragma unroll
                        for (int i = 0; i < R; ++i) {
                            auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, acc[i]);
#pragma unroll
                            for (int t = 0; t < 8; ++t) c[i][t] += (float) wc[t] * (xs[t] * d[i]);
                        }
                    }
                }
            }
            if constexpr (KS > 1) {
#pragma unroll
                for (int i = 0; i < R; ++i)
#pragma unroll
                    for (int t = 0; t < 8; ++t) part[(((size_t) ks * R + i) * 16 + ln) * 8 + t] = c[i][t];
                it.barrier(sycl::access::fence_space::local_space);
                if (ks != 0) return;
#pragma unroll
                for (int i = 0; i < R; ++i)
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        float s = 0.0f;
#pragma unroll
                        for (int k2 = 0; k2 < KS; ++k2) s += part[(((size_t) k2 * R + i) * 16 + ln) * 8 + t];
                        c[i][t] = s;
                    }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int64_t orow = (rg * R + i) * 16 + ln;
                if (orow < N) {
#pragma unroll
                    for (int t = 0; t < 8; ++t)
                        if (t < ncols) y[g * ybs + t * ycs + orow] = c[i][t];
                }
            }
        };
        const sycl::nd_range<2> nr(sycl::range<2>((size_t) G, (size_t) (nrg * KS * 16)), sycl::range<2>(1, (size_t) KS * 16));
        namespace sxo = sycl::ext::oneapi::experimental;
        h.parallel_for(nr, sxo::properties{sycl::ext::intel::experimental::grf_size<256>}, kern);
    });
}

static sycl::event xmx_launch(sycl::queue & q, int KS, int R, const void * w, int64_t N, int64_t K, int64_t G, int64_t g_div,
                              int64_t nb_tot, size_t wbb, const sycl::half * xT, const float * xdT, int ncols, float * y,
                              int64_t ycs, int64_t ybs) {
#define XL(k, r) xmx_mv<k, r>(q, w, N, K, G, g_div, nb_tot, wbb, xT, xdT, ncols, y, ycs, ybs)
#define XR(k) (R == 1 ? XL(k, 1) : R == 2 ? XL(k, 2) : XL(k, 4))
    switch (KS) {
        case 1: return XR(1);
        case 2: return XR(2);
        case 4: return XR(4);
        case 8: return XR(8);
        default: return XR(16);
    }
#undef XR
#undef XL
}

int main(int argc, char ** argv) {
    const int NCOLS = argc > 1 ? atoi(argv[1]) : 5;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("%s, %d columns\n", q.get_device().get_info<sycl::info::device::name>().c_str(), NCOLS);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> ub(0, 255);
    std::normal_distribution<float> nf(0.f, 1.f);
    struct S { const char * name; int64_t K, N, G; };
    const S shapes[] = {{"q_b 1280x32768", 1280, 32768, 1}, {"wo_b 8192x5120", 8192, 5120, 1}, {"wo_a 4096x1024 x8", 4096, 1024, 8},
                        {"shexp gu 5120x4608", 5120, 4608, 1}, {"shexp d 2304x5120", 2304, 5120, 1}, {"q_a 5120x1280", 5120, 1280, 1},
                        {"kv 5120x512", 5120, 512, 1}};
    for (const S & s : shapes) {
        const int64_t K = s.K, N = s.N, G = s.G, bpr = K / 256, nb = N * bpr;   // blocks per weight batch
        std::vector<BQ6> blocks((size_t) (G * nb));
        for (auto & b : blocks) {
            for (auto & v : b.ql) v = (uint8_t) ub(rng);
            for (auto & v : b.qh) v = (uint8_t) ub(rng);
            for (auto & v : b.sc) v = (int8_t) (ub(rng) - 128);
            b.d = fh(1e-4f * (0.5f + (float) ub(rng) / 255.0f));
        }
        // planar (per weight batch): ql | qh | sc | d
        const size_t wbb = (size_t) nb * 210;
        std::vector<uint8_t> planar((size_t) G * wbb);
        for (int64_t g = 0; g < G; ++g) {
            uint8_t * p = planar.data() + g * wbb;
            for (int64_t i = 0; i < nb; ++i) {
                const BQ6 & b = blocks[(size_t) (g * nb + i)];
                std::memcpy(p + i * 128, b.ql, 128);
                std::memcpy(p + 128 * nb + i * 64, b.qh, 64);
                std::memcpy(p + 192 * nb + i * 16, b.sc, 16);
                std::memcpy(p + 208 * nb + i * 2, &b.d, 2);
            }
        }
        std::vector<float> x((size_t) (G * NCOLS * K));
        for (auto & v : x) v = nf(rng);
        // reference (double) for a sample of rows
        std::vector<double> ref((size_t) (G * NCOLS * N), 0.0);
        std::vector<float> wrow((size_t) K);
        for (int64_t g = 0; g < G; ++g)
            for (int64_t r = 0; r < N; r += (N > 4096 ? 7 : 1)) {
                for (int64_t b = 0; b < bpr; ++b) deq_q6(blocks[(size_t) (g * nb + r * bpr + b)], wrow.data() + b * 256);
                for (int t = 0; t < NCOLS; ++t) {
                    double a = 0;
                    for (int64_t k = 0; k < K; ++k) a += (double) wrow[(size_t) k] * x[(size_t) ((g * NCOLS + t) * K + k)];
                    ref[(size_t) ((g * NCOLS + t) * N + r)] = a;
                }
            }
        // device copies: rotate over enough copies to leave L2
        const int NB = std::max(2, (int) std::min<int64_t>(16, (int64_t) (256e6 / (double) planar.size())));
        std::vector<uint8_t *> W;
        for (int i = 0; i < NB; ++i) { W.push_back(sycl::malloc_device<uint8_t>(planar.size(), q)); q.memcpy(W.back(), planar.data(), planar.size()); }
        float * dx = sycl::malloc_device<float>(x.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4);
        float * dy = sycl::malloc_device<float>((size_t) (G * NCOLS * N), q);
        sycl::half * xT = sycl::malloc_device<sycl::half>((size_t) (G * K * 8), q);
        float * fs = sycl::malloc_device<float>((size_t) (G * K / 32 * 8), q);   // xdT
        int8_t * xq = sycl::malloc_device<int8_t>((size_t) (G * NCOLS * K), q);
        float * xd = sycl::malloc_device<float>((size_t) (G * NCOLS * K / 32), q);
        q.wait();
        std::vector<float> y((size_t) (G * NCOLS * N));
        auto check = [&](const char * tag) {
            q.memcpy(y.data(), dy, y.size() * 4).wait();
            double e2 = 0, r2 = 0, emax = 0;
            for (int64_t g = 0; g < G; ++g)
                for (int64_t r = 0; r < N; r += (N > 4096 ? 7 : 1))
                    for (int t = 0; t < NCOLS; ++t) {
                        const size_t i = (size_t) ((g * NCOLS + t) * N + r);
                        const double e = y[i] - ref[i];
                        e2 += e * e; r2 += ref[i] * ref[i]; emax = std::max(emax, std::fabs(e));
                    }
            std::printf("    %-22s rel rms err %.2e (max abs %.2e)\n", tag, std::sqrt(e2 / r2), emax);
        };
        auto timeit = [&](const char * tag, auto fn) {
            for (int i = 0; i < 4; ++i) fn(W[(size_t) i % NB]);
            q.wait();
            const int reps = 100;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn(W[(size_t) i % NB]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("  %-20s %-26s %7.1f us  %5.0f GB/s\n", s.name, tag, us, (double) planar.size() / us / 1e3);
        };
        // fastmv's dp4a path (quant_x + r8m / generic, as ggml-sycl runs it)
        fastmv::Args a;
        a.K = K; a.N = N; a.G = G; a.g_div = 1; a.n_blocks_total = nb; a.w_batch_bytes = wbb;
        a.x = dx; a.x_col_stride = K; a.x_batch_stride = NCOLS * K;
        a.y = dy; a.y_col_stride = N; a.y_batch_stride = NCOLS * N; a.ncols = NCOLS; a.xq = xq; a.xd = xd;
        timeit("dp4a quant_x + mv", [&](uint8_t * w) {
            fastmv::quant_x(q, dx, K, NCOLS * K, NCOLS, G, K, xq, xd);
            a.w = w; fastmv::launch(q, fastmv::WType::Q6K_R8, a);
        });
        check("dp4a");
        timeit("dp4a mv only", [&](uint8_t * w) { a.w = w; fastmv::launch(q, fastmv::WType::Q6K_R8, a); });
        quant_xT(q, dx, K, NCOLS * K, NCOLS, G, K, xT, fs);
        for (int R : {1, 2, 4})
            for (int KS : {1, 2, 4, 8, 16}) {
                if (KS > bpr) break;
                const int64_t nrg = (N + 16 * R - 1) / (16 * R);
                if (nrg * G * KS < 256 || nrg * G * KS > 8192) continue;
                char t[64];
                std::snprintf(t, sizeof t, "xmx mv R=%d KS=%d (%lld sg)", R, KS, (long long) (nrg * G * KS));
                timeit(t, [&](uint8_t * w) { xmx_launch(q, KS, R, w, N, K, G, 1, nb, wbb, xT, fs, NCOLS, dy, N, NCOLS * N); });
                check(t);
            }
        timeit("xmx quant_xT only", [&](uint8_t *) { quant_xT(q, dx, K, NCOLS * K, NCOLS, G, K, xT, fs); });
        for (uint8_t * p : W) sycl::free(p, q);
        sycl::free(dx, q); sycl::free(dy, q); sycl::free(xT, q); sycl::free(fs, q); sycl::free(xq, q); sycl::free(xd, q);
    }
    return 0;
}
