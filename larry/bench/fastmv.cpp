// ggml-sycl/fastmv.cpp - see fastmv.hpp.
#include "fastmv.hpp"

#include <algorithm>

namespace fastmv {
namespace {

constexpr int SG = 16;

using u8x8 = sycl::vec<uint8_t, 8>;

inline float h2f(sycl::half h) { return (float) h; }

// One Q6_K unit = 32 values of block `b`: half ip (0/1), lane segment seg (0..3) -> l = 8*seg .. 8*seg+7, the values
// 128*ip + l + {0, 32, 64, 96}.  Accumulates into acc[t] for NT columns.
template <int NT>
inline void q6k_unit(const uint8_t * __restrict ql_all, const uint8_t * __restrict qh_all,
                     const int8_t * __restrict sc_all, const sycl::half * __restrict d_all, int64_t blk, int ip, int seg,
                     const float * const * xs, int64_t kbase, int nt, float * acc) {
    const uint8_t * ql = ql_all + blk * 128 + 64 * ip + 8 * seg;
    const uint8_t * qh = qh_all + blk * 64 + 32 * ip + 8 * seg;
    const u8x8 l0 = *reinterpret_cast<const u8x8 *>(ql);
    const u8x8 l1 = *reinterpret_cast<const u8x8 *>(ql + 32);
    const u8x8 hh = *reinterpret_cast<const u8x8 *>(qh);
    const int is = 8 * ip + (seg >> 1);
    const int8_t * sc = sc_all + blk * 16;
    const float d = h2f(d_all[blk]);
    const float s0 = d * (float) sc[is], s1 = d * (float) sc[is + 2], s2 = d * (float) sc[is + 4], s3 = d * (float) sc[is + 6];
    float w0[8], w1[8], w2[8], w3[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int a = l0[i], b = l1[i], h = hh[i];
        w0[i] = (float) ((int) ((a & 0xF) | ((h & 3) << 4)) - 32);
        w1[i] = (float) ((int) ((b & 0xF) | (((h >> 2) & 3) << 4)) - 32);
        w2[i] = (float) ((int) ((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
        w3[i] = (float) ((int) ((b >> 4) | (((h >> 6) & 3) << 4)) - 32);
    }
    const int64_t o = kbase + 128 * ip + 8 * seg;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t < nt) {
            const float * x = xs[t] + o;
            const sycl::float4 xa0 = *reinterpret_cast<const sycl::float4 *>(x);
            const sycl::float4 xa1 = *reinterpret_cast<const sycl::float4 *>(x + 4);
            const sycl::float4 xb0 = *reinterpret_cast<const sycl::float4 *>(x + 32);
            const sycl::float4 xb1 = *reinterpret_cast<const sycl::float4 *>(x + 36);
            const sycl::float4 xc0 = *reinterpret_cast<const sycl::float4 *>(x + 64);
            const sycl::float4 xc1 = *reinterpret_cast<const sycl::float4 *>(x + 68);
            const sycl::float4 xd0 = *reinterpret_cast<const sycl::float4 *>(x + 96);
            const sycl::float4 xd1 = *reinterpret_cast<const sycl::float4 *>(x + 100);
            float a0 = w0[0] * xa0.x() + w0[1] * xa0.y() + w0[2] * xa0.z() + w0[3] * xa0.w() + w0[4] * xa1.x() + w0[5] * xa1.y() + w0[6] * xa1.z() + w0[7] * xa1.w();
            float a1 = w1[0] * xb0.x() + w1[1] * xb0.y() + w1[2] * xb0.z() + w1[3] * xb0.w() + w1[4] * xb1.x() + w1[5] * xb1.y() + w1[6] * xb1.z() + w1[7] * xb1.w();
            float a2 = w2[0] * xc0.x() + w2[1] * xc0.y() + w2[2] * xc0.z() + w2[3] * xc0.w() + w2[4] * xc1.x() + w2[5] * xc1.y() + w2[6] * xc1.z() + w2[7] * xc1.w();
            float a3 = w3[0] * xd0.x() + w3[1] * xd0.y() + w3[2] * xd0.z() + w3[3] * xd0.w() + w3[4] * xd1.x() + w3[5] * xd1.y() + w3[6] * xd1.z() + w3[7] * xd1.w();
            acc[t] += s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        }
    }
}


inline u8x8 ld8_u16(const uint8_t * p) {   // 8 bytes from a 2-byte aligned address
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    const uint16_t a = q[0], b = q[1], c = q[2], d = q[3];
    return u8x8((uint8_t) a, (uint8_t) (a >> 8), (uint8_t) b, (uint8_t) (b >> 8), (uint8_t) c, (uint8_t) (c >> 8), (uint8_t) d, (uint8_t) (d >> 8));
}

template <int NT>
inline void q6k_unit_std(const uint8_t * __restrict blkp, int ip, int seg, const float * const * xs, int64_t kbase, int nt,
                         float * acc) {
    const uint8_t * ql = blkp + 64 * ip + 8 * seg;
    const uint8_t * qh = blkp + 128 + 32 * ip + 8 * seg;
    const u8x8 l0 = ld8_u16(ql);
    const u8x8 l1 = ld8_u16(ql + 32);
    const u8x8 hh = ld8_u16(qh);
    const int is = 8 * ip + (seg >> 1);
    const int8_t * sc = reinterpret_cast<const int8_t *>(blkp + 192);
    const float d = (float) sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t *>(blkp + 208));
    const float s0 = d * (float) sc[is], s1 = d * (float) sc[is + 2], s2 = d * (float) sc[is + 4], s3 = d * (float) sc[is + 6];
    float w0[8], w1[8], w2[8], w3[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int a = l0[i], b = l1[i], h = hh[i];
        w0[i] = (float) ((int) ((a & 0xF) | ((h & 3) << 4)) - 32);
        w1[i] = (float) ((int) ((b & 0xF) | (((h >> 2) & 3) << 4)) - 32);
        w2[i] = (float) ((int) ((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
        w3[i] = (float) ((int) ((b >> 4) | (((h >> 6) & 3) << 4)) - 32);
    }
    const int64_t o = kbase + 128 * ip + 8 * seg;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t < nt) {
            const float * x = xs[t] + o;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) { a0 += w0[i] * x[i]; a1 += w1[i] * x[32 + i]; a2 += w2[i] * x[64 + i]; a3 += w3[i] * x[96 + i]; }
            acc[t] += s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        }
    }
}

// generic kernel: work-group = RW rows x KS k-slices; each (row, k-slice) is handled by LPR lanes; a sub-group holds
// SG / LPR rows (or k-slices).  units: Q6K = 32 values (8 per 256-block); F16/F32 = 8 values.
template <WType T, int NT, int LPR>
void run(sycl::queue & q, const Args & a, int KS, int RW) {
    const int64_t K = a.K, N = a.N, G = a.G;
    const int64_t upr = (T == WType::Q6K_R || T == WType::Q6K) ? K / 32 : K / 8;     // units per row
    const int64_t bpr = K / 256;
    const int lanes_per_wg = RW * KS * LPR;                       // multiple of 16
    const int64_t n_wg_rows = (N + RW - 1) / RW;
    const sycl::range<2> gr((size_t) G, (size_t) (n_wg_rows * lanes_per_wg));
    const sycl::range<2> lr(1, (size_t) lanes_per_wg);
    const Args A = a;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (KS > 1 ? RW * KS * NT : 1)), h);
        h.parallel_for(sycl::nd_range<2>(gr, lr), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t g = (int64_t) it.get_global_id(0);
            const int lid = (int) it.get_local_id(1);
            const int grp = lid / LPR;                 // (row in wg, k-slice)
            const int lane = lid % LPR;
            const int rw = grp / KS, ks = grp % KS;
            const int64_t row = (int64_t) it.get_group(1) * RW + rw;
            const int64_t gw = g / A.g_div;
            float acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
            const float * xs[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) xs[t] = A.x + g * A.x_batch_stride + (int64_t) (t < A.ncols ? t : 0) * A.x_col_stride;
            if (row < N) {
                // this (row, k-slice)'s units: [u0, u1)
                const int64_t per = (upr + KS - 1) / KS;
                const int64_t u0 = ks * per, u1 = sycl::min(upr, u0 + per);
                if constexpr (T == WType::Q6K_R) {
                    const uint8_t * base = (const uint8_t *) A.w + gw * A.w_batch_bytes;
                    const int64_t nb = A.n_blocks_total;
                    const uint8_t * ql = base;
                    const uint8_t * qh = base + 128 * nb;
                    const int8_t * sc = (const int8_t *) (base + 192 * nb);
                    const sycl::half * dd = (const sycl::half *) (base + 208 * nb);
                    const int64_t blk0 = row * bpr;
                    for (int64_t u = u0 + lane; u < u1; u += LPR) {
                        const int64_t b = u >> 3;
                        const int r8 = (int) (u & 7);
                        q6k_unit<NT>(ql, qh, sc, dd, blk0 + b, r8 >> 2, r8 & 3, xs, b * 256, A.ncols, acc);
                    }
                } else if constexpr (T == WType::Q6K) {
                    const uint8_t * wr = (const uint8_t *) A.w + gw * A.w_batch_bytes + row * A.w_row_bytes;
                    for (int64_t u = u0 + lane; u < u1; u += LPR) {
                        const int64_t b = u >> 3;
                        const int r8 = (int) (u & 7);
                        q6k_unit_std<NT>(wr + b * 210, r8 >> 2, r8 & 3, xs, b * 256, A.ncols, acc);
                    }
                } else if constexpr (T == WType::F16) {
                    const sycl::half * wr = (const sycl::half *) ((const uint8_t *) A.w + gw * A.w_batch_bytes + row * A.w_row_bytes);
                    for (int64_t u = u0 + lane; u < u1; u += LPR) {
                        const sycl::vec<sycl::half, 8> wv = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(wr + u * 8);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            if (t < A.ncols) {
                                const sycl::float4 x0 = *reinterpret_cast<const sycl::float4 *>(xs[t] + u * 8);
                                const sycl::float4 x1 = *reinterpret_cast<const sycl::float4 *>(xs[t] + u * 8 + 4);
                                acc[t] += (float) wv[0] * x0.x() + (float) wv[1] * x0.y() + (float) wv[2] * x0.z() + (float) wv[3] * x0.w() +
                                          (float) wv[4] * x1.x() + (float) wv[5] * x1.y() + (float) wv[6] * x1.z() + (float) wv[7] * x1.w();
                            }
                        }
                    }
                } else {
                    const float * wr = (const float *) ((const uint8_t *) A.w + gw * A.w_batch_bytes + row * A.w_row_bytes);
                    for (int64_t u = u0 + lane; u < u1; u += LPR) {
                        const sycl::float4 w0 = *reinterpret_cast<const sycl::float4 *>(wr + u * 8);
                        const sycl::float4 w1 = *reinterpret_cast<const sycl::float4 *>(wr + u * 8 + 4);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            if (t < A.ncols) {
                                const sycl::float4 x0 = *reinterpret_cast<const sycl::float4 *>(xs[t] + u * 8);
                                const sycl::float4 x1 = *reinterpret_cast<const sycl::float4 *>(xs[t] + u * 8 + 4);
                                acc[t] += sycl::dot(w0, x0) + sycl::dot(w1, x1);
                            }
                        }
                    }
                }
            }
            // reduce over the LPR lanes of this (row, k-slice): they are consecutive lanes of one sub-group
            const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int off = LPR / 2; off > 0; off >>= 1) acc[t] += sycl::permute_group_by_xor(sg, acc[t], off);
            }
            if (KS == 1) {
                if (lane == 0 && row < N) {
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        if (t < A.ncols) A.y[g * A.y_batch_stride + t * A.y_col_stride + row] = acc[t];
                }
            } else {
                if (lane == 0) {
#pragma unroll
                    for (int t = 0; t < NT; ++t) part[(rw * KS + ks) * NT + t] = acc[t];
                }
                sycl::group_barrier(it.get_group());
                if (ks == 0 && lane == 0 && row < N) {
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        if (t < A.ncols) {
                            float s = 0.0f;
                            for (int k2 = 0; k2 < KS; ++k2) s += part[(rw * KS + k2) * NT + t];
                            A.y[g * A.y_batch_stride + t * A.y_col_stride + row] = s;
                        }
                    }
                }
            }
        });
    });
}

template <WType T, int NT>
void pick(sycl::queue & q, const Args & a) {
    // lanes per row from the units a row has; split K when the launch would have too few sub-groups
    const int64_t upr = (T == WType::Q6K_R || T == WType::Q6K) ? a.K / 32 : a.K / 8;
    int lpr = upr >= 128 ? 16 : upr >= 48 ? 8 : 4;
    const int64_t rows = a.N * a.G;
    const int64_t want_sg = 4096;                                  // ~2 waves of B70 hardware threads
    int ks = 1;
    while (ks < 64 && rows * ks * lpr / SG < want_sg && upr / (ks * 2) >= lpr * 2) ks *= 2;
    int rw = std::max(1, std::min(256 / (ks * lpr), 16));           // rows per work-group (wg <= 256 lanes)
    if (ks * lpr > 256) rw = 1;
    // a work-group must be a whole number of sub-groups
    while ((rw * ks * lpr) % SG) ++rw;
    if (lpr == 16) run<T, NT, 16>(q, a, ks, rw);
    else if (lpr == 8) run<T, NT, 8>(q, a, ks, rw);
    else run<T, NT, 4>(q, a, ks, rw);
}

}  // namespace

bool launch(sycl::queue & q, WType t, const Args & a) {
    if (a.ncols < 1 || a.ncols > 8 || a.K <= 0 || a.N <= 0) return false;
    if ((t == WType::Q6K_R || t == WType::Q6K) && a.K % 256) return false;
    if (a.K % 8) return false;
    const bool one = a.ncols == 1;
    switch (t) {
        case WType::Q6K_R: one ? pick<WType::Q6K_R, 1>(q, a) : pick<WType::Q6K_R, 8>(q, a); break;
        case WType::Q6K: one ? pick<WType::Q6K, 1>(q, a) : pick<WType::Q6K, 8>(q, a); break;
        case WType::F16: one ? pick<WType::F16, 1>(q, a) : pick<WType::F16, 8>(q, a); break;
        case WType::F32: one ? pick<WType::F32, 1>(q, a) : pick<WType::F32, 8>(q, a); break;
    }
    return true;
}

}  // namespace fastmv
