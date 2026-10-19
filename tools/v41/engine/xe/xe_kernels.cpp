// tools/v41/engine/xe/xe_kernels.cpp - see xe_kernels.hpp.
#include "xe_kernels.hpp"

#include <cstddef>

namespace strata::v41::xe {
namespace {

constexpr int SG = 16;          ///< sub-group size (Xe2 SIMD16)
constexpr int RPW = 8;          ///< weight rows (sub-groups) per work-group

inline float h2f(const uint8_t* p) {
    const uint16_t u = (uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8));
    return (float) sycl::bit_cast<sycl::half>(u);
}

inline uint32_t ld32(const uint8_t* p) {   // unaligned little-endian
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

// digit n (0..4, most significant first) of a PTQ1_0 byte, as -1 / 0 / +1
template <int N>
inline float trit_m1(uint32_t b) {
    constexpr uint32_t p = N == 0 ? 1u : N == 1 ? 3u : N == 2 ? 9u : N == 3 ? 27u : 81u;
    return (float) ((int) ((((b * p) & 0xFFu) * 3u) >> 8) - 1);
}

// ---- per-lane partial dot products of R weight rows with NT activation rows xs[t] (n floats each) ----
// The activations a lane needs for one block are loaded ONCE into registers and applied to all R rows (the rows'
// x traffic was the bound: one row per sub-group read 16x more activation bytes than weight bytes).
// acc is [R][NT].

template <int NT, int R>
inline void rows_q20(const uint8_t* const* rows, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 64;
    const int q = lane & 3;
    for (int b = lane >> 2; b < nb; b += 4) {
        const int base = b * 64 + q * 16;
        float x[NT][16];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const sycl::float4* x4 = reinterpret_cast<const sycl::float4*>(xs[t] + base);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const sycl::float4 xv = x4[v];
                    x[t][4 * v] = xv.x(); x[t][4 * v + 1] = xv.y(); x[t][4 * v + 2] = xv.z(); x[t][4 * v + 3] = xv.w();
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = rows[r] + (size_t) b * 18;
            const float d = h2f(blk);
            const uint32_t w = ld32(blk + 2 + q * 4);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) s += (float) ((int) ((w >> (2 * i)) & 3u) - 1) * x[t][i];
                    acc[r * NT + t] += d * s;
                }
            }
        }
    }
}

template <int NT, int R>
inline void rows_ptq1(const uint8_t* const* rows, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 128;
    const bool hb = lane < 8, hc = lane < 2;
    for (int b = 0; b < nb; ++b) {
        const int base = b * 128;
        float xa[NT][5], xb[NT][5], xc[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const float* x = xs[t] + base;
#pragma unroll
                for (int k = 0; k < 5; ++k) xa[t][k] = x[lane + 16 * k];
#pragma unroll
                for (int k = 0; k < 5; ++k) xb[t][k] = hb ? x[80 + lane + 8 * k] : 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) xc[t][k] = hc ? x[120 + lane + 2 * k] : 0.f;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = rows[r] + (size_t) b * 28;
            const float d = h2f(blk + 26);
            const uint32_t b0 = blk[lane];
            const uint32_t b1 = hb ? blk[16 + lane] : 0u;
            const uint32_t b2 = hc ? blk[24 + lane] : 0u;
            float wa[5], wb[5], wc[4];
            wa[0] = trit_m1<0>(b0); wa[1] = trit_m1<1>(b0); wa[2] = trit_m1<2>(b0); wa[3] = trit_m1<3>(b0); wa[4] = trit_m1<4>(b0);
            wb[0] = trit_m1<0>(b1); wb[1] = trit_m1<1>(b1); wb[2] = trit_m1<2>(b1); wb[3] = trit_m1<3>(b1); wb[4] = trit_m1<4>(b1);
            wc[0] = trit_m1<0>(b2); wc[1] = trit_m1<1>(b2); wc[2] = trit_m1<2>(b2); wc[3] = trit_m1<3>(b2);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    float s = 0.f;
#pragma unroll
                    for (int k = 0; k < 5; ++k) s += wa[k] * xa[t][k] + wb[k] * xb[t][k];
#pragma unroll
                    for (int k = 0; k < 4; ++k) s += wc[k] * xc[t][k];
                    acc[r * NT + t] += d * s;
                }
            }
        }
    }
}

template <int NT, int R>
inline void rows_q3k(const uint8_t* const* rows, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 256;
    for (int b = 0; b < nb; ++b) {
        const int base = b * 256;
        float x[NT][16];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const float* xp = xs[t] + base;
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int o = h * 128 + j * 32 + lane;
                        x[t][(h * 4 + j) * 2] = xp[o];
                        x[t][(h * 4 + j) * 2 + 1] = xp[o + 16];
                    }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = rows[r] + (size_t) b * 110;
            const uint8_t* qs = blk + 32;
            const uint8_t* sc = blk + 96;
            const float dall = h2f(blk + 108);
            const uint32_t a0 = ld32(sc), a1 = ld32(sc + 4), tmp = ld32(sc + 8);
            const uint32_t s4[4] = {
                (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4),
                (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4),
                ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4),
                ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4),
            };
            const uint32_t hlo = blk[lane], hhi = blk[lane + 16];
            float wf[16];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t qlo = qs[h * 32 + lane], qhi = qs[h * 32 + lane + 16];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int is = h * 8 + 2 * j;
                    const float slo = (float) ((int) ((s4[is >> 2] >> (8 * (is & 3))) & 0xffu) - 32);
                    const float shi = (float) ((int) ((s4[(is + 1) >> 2] >> (8 * ((is + 1) & 3))) & 0xffu) - 32);
                    const int mbit = h * 4 + j;
                    const int vlo = (int) ((qlo >> (2 * j)) & 3u) - (((hlo >> mbit) & 1u) ? 0 : 4);
                    const int vhi = (int) ((qhi >> (2 * j)) & 3u) - (((hhi >> mbit) & 1u) ? 0 : 4);
                    wf[(h * 4 + j) * 2] = slo * (float) vlo;
                    wf[(h * 4 + j) * 2 + 1] = shi * (float) vhi;
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) s += wf[i] * x[t][i];
                    acc[r * NT + t] += dall * s;
                }
            }
        }
    }
}

// ---- the multi-token form (NT up to 8): ONE row, its block decoded once, then each token's x loaded and applied.
// Holding NT tokens' activations in registers (the R-row form) spills at NT = 8.

template <int NT>
inline void row_q20(const uint8_t* row, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 64;
    const int q = lane & 3;
    for (int b = lane >> 2; b < nb; b += 4) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        const uint32_t w = ld32(blk + 2 + q * 4);
        float wf[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) wf[i] = (float) ((int) ((w >> (2 * i)) & 3u) - 1);
        const int base = b * 64 + q * 16;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const sycl::float4* x4 = reinterpret_cast<const sycl::float4*>(xs[t] + base);
                float s = 0.f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const sycl::float4 x = x4[v];
                    s += wf[4 * v] * x.x() + wf[4 * v + 1] * x.y() + wf[4 * v + 2] * x.z() + wf[4 * v + 3] * x.w();
                }
                acc[t] += d * s;
            }
        }
    }
}

template <int NT>
inline void row_ptq1(const uint8_t* row, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 128;
    const bool hb = lane < 8, hc = lane < 2;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 28;
        const float d = h2f(blk + 26);
        const int base = b * 128;
        const uint32_t b0 = blk[lane];
        const uint32_t b1 = hb ? blk[16 + lane] : 0u;
        const uint32_t b2 = hc ? blk[24 + lane] : 0u;
        float wa[5], wb[5], wc[4];
        wa[0] = trit_m1<0>(b0); wa[1] = trit_m1<1>(b0); wa[2] = trit_m1<2>(b0); wa[3] = trit_m1<3>(b0); wa[4] = trit_m1<4>(b0);
        wb[0] = trit_m1<0>(b1); wb[1] = trit_m1<1>(b1); wb[2] = trit_m1<2>(b1); wb[3] = trit_m1<3>(b1); wb[4] = trit_m1<4>(b1);
        wc[0] = trit_m1<0>(b2); wc[1] = trit_m1<1>(b2); wc[2] = trit_m1<2>(b2); wc[3] = trit_m1<3>(b2);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const float* x = xs[t] + base;
                float s = 0.f;
#pragma unroll
                for (int k = 0; k < 5; ++k) s += wa[k] * x[lane + 16 * k];
                if (hb) {
#pragma unroll
                    for (int k = 0; k < 5; ++k) s += wb[k] * x[80 + lane + 8 * k];
                }
                if (hc) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) s += wc[k] * x[120 + lane + 2 * k];
                }
                acc[t] += d * s;
            }
        }
    }
}

template <int NT>
inline void row_q3k(const uint8_t* row, int n, const float* const* xs, int nt, float* acc, int lane) {
    const int nb = n / 256;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 110;
        const uint8_t* qs = blk + 32;
        const uint8_t* sc = blk + 96;
        const float dall = h2f(blk + 108);
        const uint32_t a0 = ld32(sc), a1 = ld32(sc + 4), tmp = ld32(sc + 8);
        const uint32_t s4[4] = {
            (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4),
            (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4),
            ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4),
            ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4),
        };
        const uint32_t hlo = blk[lane], hhi = blk[lane + 16];
        float wf[16];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t qlo = qs[h * 32 + lane], qhi = qs[h * 32 + lane + 16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int is = h * 8 + 2 * j;
                const float slo = (float) ((int) ((s4[is >> 2] >> (8 * (is & 3))) & 0xffu) - 32);
                const float shi = (float) ((int) ((s4[(is + 1) >> 2] >> (8 * ((is + 1) & 3))) & 0xffu) - 32);
                const int mbit = h * 4 + j;
                const int vlo = (int) ((qlo >> (2 * j)) & 3u) - (((hlo >> mbit) & 1u) ? 0 : 4);
                const int vhi = (int) ((qhi >> (2 * j)) & 3u) - (((hhi >> mbit) & 1u) ? 0 : 4);
                wf[(h * 4 + j) * 2] = dall * slo * (float) vlo;
                wf[(h * 4 + j) * 2 + 1] = dall * shi * (float) vhi;
            }
        }
        const int base = b * 256;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const float* x = xs[t] + base;
                float s = 0.f;
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int o = h * 128 + j * 32 + lane;
                        s += wf[(h * 4 + j) * 2] * x[o] + wf[(h * 4 + j) * 2 + 1] * x[o + 16];
                    }
                acc[t] += s;
            }
        }
    }
}

// R rows for one token (R-row register reuse), or one row for up to 8 tokens
template <int NT, int R>
inline void rows_dot(int type, const uint8_t* const* rows, int n, const float* const* xs, int nt, float* acc, int lane) {
    if constexpr (NT == 1) {
        if (type == kQ20) rows_q20<1, R>(rows, n, xs, nt, acc, lane);
        else if (type == kPtq1) rows_ptq1<1, R>(rows, n, xs, nt, acc, lane);
        else rows_q3k<1, R>(rows, n, xs, nt, acc, lane);
    } else {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (type == kQ20) row_q20<NT>(rows[r], n, xs, nt, acc + r * NT, lane);
            else if (type == kPtq1) row_ptq1<NT>(rows[r], n, xs, nt, acc + r * NT, lane);
            else row_q3k<NT>(rows[r], n, xs, nt, acc + r * NT, lane);
        }
    }
}

// gate_up: a sub-group computes RG consecutive rows of gate AND of up (2*RG row dots sharing the x loads)
template <int NT, int RG>
sycl::event gate_up(sycl::queue& q, const XeJob* jobs, int nj, const float* X, float* H, int n_embd, int n_ff,
                    float L) {
    const int rows_wg = RPW * RG;
    const int rb = n_ff / rows_wg;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * rb * RPW * SG), sycl::range<1>(RPW * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int wg = (int) it.get_group(0);
        const int j = wg / rb;
        const sycl::sub_group sg = it.get_sub_group();
        const int r0 = (wg % rb) * rows_wg + (int) sg.get_group_linear_id() * RG;
        const int lane = (int) sg.get_local_linear_id();
        const XeJob& jb = jobs[j];
        const int nt = jb.nt;
        const float* xs[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) xs[t] = X + (size_t) jb.tok[t < nt ? t : 0] * n_embd;
        const uint8_t* rows[2 * RG];
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            rows[r] = jb.blob + (size_t) (r0 + r) * jb.gu_row;
            rows[RG + r] = jb.blob + jb.up_off + (size_t) (r0 + r) * jb.gu_row;
        }
        float acc[2 * RG * NT];
#pragma unroll
        for (int i = 0; i < 2 * RG * NT; ++i) acc[i] = 0.f;
        rows_dot<NT, 2 * RG>(jb.gu_type, rows, n_embd, xs, nt, acc, lane);
#pragma unroll
        for (int r = 0; r < RG; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    float g = sycl::reduce_over_group(sg, acc[r * NT + t], sycl::plus<float>());
                    float u = sycl::reduce_over_group(sg, acc[(RG + r) * NT + t], sycl::plus<float>());
                    if (lane == 0) {
                        g = sycl::fmin(g, L);
                        u = sycl::fmax(-L, sycl::fmin(u, L));
                        H[((size_t) j * kMaxJobTok + t) * n_ff + r0 + r] = g / (1.0f + sycl::exp(-g)) * u;
                    }
                }
            }
    });
}

// down: a sub-group computes RD consecutive output rows for every job (weighted sum over the jobs)
template <int NT, int RD>
sycl::event down(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const float* H, float* OUT, int n_embd,
                 int n_ff, bool accumulate) {
    const int rows_wg = RPW * RD;
    const int rb = n_embd / rows_wg;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) rb * RPW * SG), sycl::range<1>(RPW * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int i0 = (int) it.get_group(0) * rows_wg + (int) sg.get_group_linear_id() * RD;
        const int lane = (int) sg.get_local_linear_id();
        float o[RD * NT];
#pragma unroll
        for (int k = 0; k < RD * NT; ++k) o[k] = 0.f;
        for (int j = 0; j < nj; ++j) {
            const XeJob& jb = jobs[j];
            const int nt = jb.nt;
            const float* hs[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) hs[t] = H + ((size_t) j * kMaxJobTok + (t < nt ? t : 0)) * n_ff;
            const uint8_t* rows[RD];
#pragma unroll
            for (int r = 0; r < RD; ++r) rows[r] = jb.blob + jb.down_off + (size_t) (i0 + r) * jb.d_row;
            float a[RD * NT];
#pragma unroll
            for (int k = 0; k < RD * NT; ++k) a[k] = 0.f;
            rows_dot<NT, RD>(jb.d_type, rows, n_ff, hs, nt, a, lane);
#pragma unroll
            for (int r = 0; r < RD; ++r)
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    if (s < nt) {
                        const float v = sycl::reduce_over_group(sg, a[r * NT + s], sycl::plus<float>()) * jb.w[s];
                        const int tk = jb.tok[s];
#pragma unroll
                        for (int t = 0; t < NT; ++t) if (t == tk) o[r * NT + t] += v;
                    }
                }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < RD; ++r)
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    if (t < ntok) {
                        float* p = OUT + (size_t) t * n_embd + i0 + r;
                        *p = accumulate ? *p + o[r * NT + t] : o[r * NT + t];
                    }
        }
    });
}

sycl::event down_y(sycl::queue& q, const XeJob* jobs, int nj, const float* H, float* Y, int n_embd, int n_ff) {
    const int rb = n_embd / RPW;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * rb * RPW * SG), sycl::range<1>(RPW * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int wg = (int) it.get_group(0);
        const int j = wg / rb;
        const sycl::sub_group sg = it.get_sub_group();
        const int i = (wg % rb) * RPW + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const XeJob& jb = jobs[j];
        const int nt = jb.nt;
        const float* hs[kMaxJobTok];
#pragma unroll
        for (int t = 0; t < kMaxJobTok; ++t) hs[t] = H + ((size_t) j * kMaxJobTok + (t < nt ? t : 0)) * n_ff;
        const uint8_t* rows[1] = {jb.blob + jb.down_off + (size_t) i * jb.d_row};
        float a[kMaxJobTok];
#pragma unroll
        for (int t = 0; t < kMaxJobTok; ++t) a[t] = 0.f;
        rows_dot<kMaxJobTok, 1>(jb.d_type, rows, n_ff, hs, nt, a, lane);
#pragma unroll
        for (int s = 0; s < kMaxJobTok; ++s) {
            if (s < nt) {
                const float v = sycl::reduce_over_group(sg, a[s], sycl::plus<float>());
                if (lane == 0) Y[((size_t) j * kMaxJobTok + s) * n_embd + i] = v;
            }
        }
    });
}

}  // namespace

sycl::event xe_experts(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const float* X, float* H, float* OUT,
                       int n_embd, int n_ff, float swiglu_limit, bool accumulate) {
    if (ntok <= 1) {
        gate_up<1, 2>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
        return down<1, 4>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    gate_up<kMaxJobTok, 1>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
    return down<kMaxJobTok, 1>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
}

sycl::event xe_experts_y(sycl::queue& q, const XeJob* jobs, int nj, const float* X, float* H, float* Y, int n_embd,
                         int n_ff, float swiglu_limit) {
    gate_up<kMaxJobTok, 1>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
    return down_y(q, jobs, nj, H, Y, n_embd, n_ff);
}

sycl::event xe_gather(sycl::queue& q, const float* Y, const int32_t* S, const float* W, int ntok, int top_k,
                      int n_embd, float* OUT, bool accumulate) {
    const size_t n = (size_t) ntok * n_embd;
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t g = id[0];
        const int t = (int) (g / n_embd), i = (int) (g % n_embd);
        float s = accumulate ? OUT[g] : 0.f;
        for (int k = 0; k < top_k; ++k) {
            const int32_t sl = S[t * top_k + k];
            if (sl >= 0) s += W[t * top_k + k] * Y[(size_t) sl * n_embd + i];
        }
        OUT[g] = s;
    });
}

sycl::event xe_sum_parts(sycl::queue& q, const float* parts, int nparts, int n, float* out, bool accumulate) {
    return q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        float s = accumulate ? out[i] : 0.f;
        for (int k = 0; k < nparts; ++k) s += parts[(size_t) k * n + i];
        out[i] = s;
    });
}

namespace {

inline float dequant_one(int type, const uint8_t* row, int v) {
    if (type == kQ20) {
        const uint8_t* blk = row + (size_t) (v / 64) * 18;
        const int j = v % 64;
        return (float) ((int) ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * h2f(blk);
    }
    if (type == kPtq1) {
        const uint8_t* blk = row + (size_t) (v / 128) * 28;
        const int j = v % 128;
        int byte, dig;
        if (j < 80) { byte = j % 16; dig = j / 16; }
        else if (j < 120) { byte = 16 + (j - 80) % 8; dig = (j - 80) / 8; }
        else { byte = 24 + (j - 120) % 2; dig = (j - 120) / 2; }
        const uint32_t p = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
        const int t = (int) ((((blk[byte] * p) & 0xFFu) * 3u) >> 8);
        return (float) (t - 1) * h2f(blk + 26);
    }
    // Q3_K, ggml's dequantize_row_q3_K
    const uint8_t* blk = row + (size_t) (v / 256) * 110;
    const int j256 = v % 256;
    const int h = j256 / 128, jj = (j256 % 128) / 32, l = j256 % 32;
    const uint8_t* sc = blk + 96;
    const uint32_t a0 = ld32(sc), a1 = ld32(sc + 4), tmp = ld32(sc + 8);
    const uint32_t s4[4] = {
        (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4),
        (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4),
        ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4),
        ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4),
    };
    const int is = h * 8 + 2 * jj + (l >= 16 ? 1 : 0);
    const float scl = (float) ((int) ((s4[is >> 2] >> (8 * (is & 3))) & 0xffu) - 32);
    const uint32_t qb = blk[32 + h * 32 + l];
    const uint32_t hm = blk[l];
    const int qv = (int) ((qb >> (2 * jj)) & 3u) - (((hm >> (h * 4 + jj)) & 1u) ? 0 : 4);
    return h2f(blk + 108) * scl * (float) qv;
}

}  // namespace

sycl::event xe_dequant_rows(sycl::queue& q, int type, const uint8_t* rows, int nrows, int row_bytes, int n, float* y) {
    return q.parallel_for(sycl::range<1>((size_t) nrows * n), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / n), v = (int) (id[0] % n);
        y[id[0]] = dequant_one(type, rows + (size_t) r * row_bytes, v);
    });
}

// one work-item per 8 consecutive output values (consecutive work-items write consecutive 16-byte chunks)
sycl::event xe_dequant_rows_f16(sycl::queue& q, int type, const uint8_t* rows, int nrows, int row_bytes, int n, sycl::half* y) {
    using h8 = sycl::vec<sycl::half, 8>;
    const int per_row = n / 8;
    const size_t total = (size_t) nrows * per_row;
    return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / per_row);
        const int v0 = (int) (id[0] % per_row) * 8;
        const uint8_t* row = rows + (size_t) r * row_bytes;
        h8 o;
        if (type == kQ20) {
            const uint8_t* blk = row + (size_t) (v0 / 64) * 18;
            const float d = h2f(blk);
            const int j = v0 % 64;
            const uint32_t w = (uint32_t) blk[2 + j / 4] | ((uint32_t) blk[3 + j / 4] << 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = (sycl::half) ((float) ((int) ((w >> (2 * i)) & 3u) - 1) * d);
        } else if (type == kPtq1) {
            const uint8_t* blk = row + (size_t) (v0 / 128) * 28;
            const float d = h2f(blk + 26);
            const int j = v0 % 128;
            if (j < 80) {   // bytes j%16 .. +7, digit j/16
                const int m0 = j % 16, dg = j / 16;
                const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : dg == 3 ? 27u : 81u;
#pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = (sycl::half) ((float) ((int) ((((blk[m0 + i] * p) & 0xFFu) * 3u) >> 8) - 1) * d);
            } else if (j < 120) {   // bytes 16..23, digit (j-80)/8
                const int dg = (j - 80) / 8;
                const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : dg == 3 ? 27u : 81u;
#pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = (sycl::half) ((float) ((int) ((((blk[16 + i] * p) & 0xFFu) * 3u) >> 8) - 1) * d);
            } else {   // values 120..127 = 120 + h + 2n: bytes 24/25, digits 0..3
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int h = i % 2, dg = i / 2;
                    const uint32_t p = dg == 0 ? 1u : dg == 1 ? 3u : dg == 2 ? 9u : 27u;
                    o[i] = (sycl::half) ((float) ((int) ((((blk[24 + h] * p) & 0xFFu) * 3u) >> 8) - 1) * d);
                }
            }
        } else {   // Q3_K: 8 values of one 32-group (h, j), one half of it (one 6-bit scale)
            const uint8_t* blk = row + (size_t) (v0 / 256) * 110;
            const int j256 = v0 % 256;
            const int h = j256 / 128, jj = (j256 % 128) / 32, l0 = j256 % 32;
            const uint8_t* sc = blk + 96;
            const int is = h * 8 + 2 * jj + (l0 >= 16 ? 1 : 0);
            // the 6-bit scale `is`: low 4 bits from byte is%8 (nibble is/8), high 2 bits from byte 8 + is%4 (pair is/4)
            const uint32_t lo4 = (sc[is % 8] >> (4 * (is / 8))) & 0x0fu;
            const uint32_t hi2 = (sc[8 + is % 4] >> (2 * (is / 4))) & 0x03u;
            const float dl = h2f(blk + 108) * (float) ((int) (lo4 | (hi2 << 4)) - 32);
            const int mbit = h * 4 + jj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int l = l0 + i;
                const int qv = (int) ((blk[32 + h * 32 + l] >> (2 * jj)) & 3u) - (((blk[l] >> mbit) & 1u) ? 0 : 4);
                o[i] = (sycl::half) (dl * (float) qv);
            }
        }
        *reinterpret_cast<h8*>(y + (size_t) r * n + v0) = o;
    });
}

sycl::event xe_gather_rows_f16(sycl::queue& q, const float* X, const int32_t* tok, int m, int n, sycl::half* Xg) {
    const size_t per_row = (size_t) n / 4;
    return q.parallel_for(sycl::range<1>((size_t) m * per_row), [=](sycl::id<1> id) {
        const int i = (int) (id[0] / per_row), c = (int) (id[0] % per_row) * 4;
        const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(X + (size_t) tok[i] * n + c);
        sycl::vec<sycl::half, 4> o((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        *reinterpret_cast<sycl::vec<sycl::half, 4>*>(Xg + (size_t) i * n + c) = o;
    });
}

sycl::event xe_swiglu_f16(sycl::queue& q, const float* G, int m, int F, float L, sycl::half* Hh) {
    return q.parallel_for(sycl::range<1>((size_t) m * F), [=](sycl::id<1> id) {
        const int i = (int) (id[0] / F), r = (int) (id[0] % F);
        const float g = sycl::fmin(G[(size_t) i * 2 * F + r], L);
        const float u = sycl::fmax(-L, sycl::fmin(G[(size_t) i * 2 * F + F + r], L));
        Hh[id[0]] = (sycl::half) (g / (1.0f + sycl::exp(-g)) * u);
    });
}

}  // namespace strata::v41::xe
