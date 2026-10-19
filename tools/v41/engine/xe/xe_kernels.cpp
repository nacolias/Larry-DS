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

// ---- per-lane partial dot products of one weight row with NT activation rows xs[t] (n floats each) ----

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
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 28;
        const float d = h2f(blk + 26);
        const int base = b * 128;
        const uint32_t b0 = blk[lane];
        float wa[5];
        wa[0] = trit_m1<0>(b0); wa[1] = trit_m1<1>(b0); wa[2] = trit_m1<2>(b0); wa[3] = trit_m1<3>(b0); wa[4] = trit_m1<4>(b0);
        float wb[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
        float wc[4] = {0.f, 0.f, 0.f, 0.f};
        const bool hb = lane < 8, hc = lane < 2;
        if (hb) {
            const uint32_t b1 = blk[16 + lane];
            wb[0] = trit_m1<0>(b1); wb[1] = trit_m1<1>(b1); wb[2] = trit_m1<2>(b1); wb[3] = trit_m1<3>(b1); wb[4] = trit_m1<4>(b1);
        }
        if (hc) {
            const uint32_t b2 = blk[24 + lane];
            wc[0] = trit_m1<0>(b2); wc[1] = trit_m1<1>(b2); wc[2] = trit_m1<2>(b2); wc[3] = trit_m1<3>(b2);
        }
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
        float wf[16];   // [h][j][lo/hi]
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t qlo = qs[h * 32 + lane], qhi = qs[h * 32 + lane + 16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int is = h * 8 + 2 * j;   // the lo half's scale; the hi half's is is + 1
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

template <int NT>
inline void row_dot(int type, const uint8_t* row, int n, const float* const* xs, int nt, float* acc, int lane) {
    if (type == kQ20) row_q20<NT>(row, n, xs, nt, acc, lane);
    else if (type == kPtq1) row_ptq1<NT>(row, n, xs, nt, acc, lane);
    else row_q3k<NT>(row, n, xs, nt, acc, lane);
}

template <int NT>
sycl::event gate_up(sycl::queue& q, const XeJob* jobs, int nj, const float* X, float* H, int n_embd, int n_ff,
                    float L) {
    const int rb = n_ff / RPW;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * rb * RPW * SG), sycl::range<1>(RPW * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int wg = (int) it.get_group(0);
        const int j = wg / rb;
        const sycl::sub_group sg = it.get_sub_group();
        const int r = (wg % rb) * RPW + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const XeJob& jb = jobs[j];
        const int nt = jb.nt;
        const float* xs[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) xs[t] = X + (size_t) jb.tok[t < nt ? t : 0] * n_embd;
        float ag[NT], au[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) { ag[t] = 0.f; au[t] = 0.f; }
        row_dot<NT>(jb.gu_type, jb.blob + (size_t) r * jb.gu_row, n_embd, xs, nt, ag, lane);
        row_dot<NT>(jb.gu_type, jb.blob + jb.up_off + (size_t) r * jb.gu_row, n_embd, xs, nt, au, lane);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                float g = sycl::reduce_over_group(sg, ag[t], sycl::plus<float>());
                float u = sycl::reduce_over_group(sg, au[t], sycl::plus<float>());
                if (lane == 0) {
                    g = sycl::fmin(g, L);
                    u = sycl::fmax(-L, sycl::fmin(u, L));
                    H[((size_t) j * kMaxJobTok + t) * n_ff + r] = g / (1.0f + sycl::exp(-g)) * u;
                }
            }
        }
    });
}

template <int NT>
sycl::event down(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const float* H, float* OUT, int n_embd,
                 int n_ff, bool accumulate) {
    const int rb = n_embd / RPW;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) rb * RPW * SG), sycl::range<1>(RPW * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int i = (int) it.get_group(0) * RPW + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        float o[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) o[t] = 0.f;
        for (int j = 0; j < nj; ++j) {
            const XeJob& jb = jobs[j];
            const int nt = jb.nt;
            const float* hs[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) hs[t] = H + ((size_t) j * kMaxJobTok + (t < nt ? t : 0)) * n_ff;
            float a[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) a[t] = 0.f;
            row_dot<NT>(jb.d_type, jb.blob + jb.down_off + (size_t) i * jb.d_row, n_ff, hs, nt, a, lane);
#pragma unroll
            for (int s = 0; s < NT; ++s) {
                if (s < nt) {
                    const float v = sycl::reduce_over_group(sg, a[s], sycl::plus<float>()) * jb.w[s];
                    const int tk = jb.tok[s];
#pragma unroll
                    for (int t = 0; t < NT; ++t) if (t == tk) o[t] += v;
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t < ntok) {
                    float* p = OUT + (size_t) t * n_embd + i;
                    *p = accumulate ? *p + o[t] : o[t];
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
        float a[kMaxJobTok];
#pragma unroll
        for (int t = 0; t < kMaxJobTok; ++t) a[t] = 0.f;
        row_dot<kMaxJobTok>(jb.d_type, jb.blob + jb.down_off + (size_t) i * jb.d_row, n_ff, hs, nt, a, lane);
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
        gate_up<1>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
        return down<1>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    gate_up<kMaxJobTok>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
    return down<kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
}

sycl::event xe_experts_y(sycl::queue& q, const XeJob* jobs, int nj, const float* X, float* H, float* Y, int n_embd,
                         int n_ff, float swiglu_limit) {
    gate_up<kMaxJobTok>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
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

sycl::event xe_dequant_rows(sycl::queue& q, int type, const uint8_t* rows, int nrows, int row_bytes, int n, float* y) {
    return q.parallel_for(sycl::range<1>((size_t) nrows * n), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / n), v = (int) (id[0] % n);
        const uint8_t* row = rows + (size_t) r * row_bytes;
        float out = 0.f;
        if (type == kQ20) {
            const uint8_t* blk = row + (size_t) (v / 64) * 18;
            const int j = v % 64;
            out = (float) ((int) ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * h2f(blk);
        } else if (type == kPtq1) {
            const uint8_t* blk = row + (size_t) (v / 128) * 28;
            const int j = v % 128;
            int byte, dig;
            if (j < 80) { byte = j % 16; dig = j / 16; }
            else if (j < 120) { byte = 16 + (j - 80) % 8; dig = (j - 80) / 8; }
            else { byte = 24 + (j - 120) % 2; dig = (j - 120) / 2; }
            const uint32_t p = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
            const int t = (int) ((((blk[byte] * p) & 0xFFu) * 3u) >> 8);
            out = (float) (t - 1) * h2f(blk + 26);
        } else {   // Q3_K, ggml's dequantize_row_q3_K
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
            out = h2f(blk + 108) * scl * (float) qv;
        }
        y[id[0]] = out;
    });
}

}  // namespace strata::v41::xe
