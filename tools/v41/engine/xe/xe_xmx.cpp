// tools/v41/engine/xe/xe_xmx.cpp - larry: decode / verify routed experts on the B70's XMX engines (fp16 DPAS).
//
// The dp4a kernels (xe_planar.cpp) re-read a token's int8 activations for every weight row, so a call's cost grows with
// the tokens a job has: 3 experts 66 us for 1 token, 88 for 2, 115 for 3, 170 for 5 (test_ds41_xe --bench-q8) - the
// shape of every DSpark verify pass.  Here a sub-group takes 16 weight rows (B: lane n decodes row n, 16 K values a
// step) against the call's 8 token rows at once (A: fp16 activations stored [K][8], a lane loads its K index's 8
// values), so the weights stream once whatever the tokens.
//   * B holds small exact integers (Q2_0 code - 1, PTQ1 trit - 1, Q3_K q * (6-bit scale - 32), |.| <= 128); the block's
//     fp16 scale multiplies the block's fp32 accumulator per lane afterwards - the products are exact.
//   * Activations stay fp16 (no 8-bit rounding as in the dp4a path): closer to the f32 reference.
//   * gate/up: a work-group = KS sub-groups on the same 16 rows of one job, each a quarter of K, summed in local memory;
//     down: a work-group = KS sub-groups on 16 output rows, every job in turn, summed the same way.
#include "xe_kernels.hpp"

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace strata::v41::xe {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace sxo = sycl::ext::oneapi::experimental;
using h2 = sycl::vec<sycl::half, 2>;
using Acc = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;
using MatA = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major>;
using MatB = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major>;
constexpr int SG = 16;

inline float hf(uint16_t u) { return (float) sycl::bit_cast<sycl::half>(u); }

/// the 4 bytes of x as halves (byte - off + 1024) * s, in byte order (off is 1024 + the bytes' bias)
inline void bytes4(uint32_t x, h2 off, h2 s2, sycl::half* o) {
    const h2 lo = (sycl::bit_cast<h2>((x & 0x00FF00FFu) | 0x64006400u) - off) * s2;          // bytes 0, 2
    const h2 hi = (sycl::bit_cast<h2>(((x >> 8) & 0x00FF00FFu) | 0x64006400u) - off) * s2;   // bytes 1, 3
    o[0] = lo[0]; o[1] = hi[0]; o[2] = lo[1]; o[3] = hi[1];
}

/// PTQ1_0: F_d of 4 bytes at multiplier p = 3^(d+1) (byte i = floor(b_i p / 256)), and trit d = F_d - 3 F_(d-1) (0..2)
inline uint32_t fword(uint32_t W, uint32_t p) {
    const uint32_t lo = W & 0x00FF00FFu, hi = (W >> 8) & 0x00FF00FFu;
    return (((lo * p) >> 8) & 0x00FF00FFu) | ((hi * p) & 0xFF00FF00u);
}
inline uint32_t trits(uint32_t W, int d) {
    constexpr uint32_t P[6] = {1u, 3u, 9u, 27u, 81u, 243u};
    const uint32_t Fd = fword(W, P[d + 1]);
    return d > 0 ? Fd - 3u * fword(W, P[d]) : Fd;   // per byte: F_d >= 3 F_(d-1), no borrow
}

/// One activation step: A = xt[k .. k + 15][0..7] (lane ln: K index k + ln)
inline void load_a(sycl::sub_group sg, MatA& ma, const sycl::half* xt, int k, int ln) {
    const sycl::vec<sycl::half, 8> a8 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xt + (size_t) (k + ln) * 8);
    auto wa = sycl::ext::oneapi::detail::get_wi_data(sg, ma);
#pragma unroll
    for (int r = 0; r < 8; ++r) wa[r] = a8[r];
}

inline void put_b(sycl::sub_group sg, MatB& mb, const sycl::half (&v)[16]) {
    auto wb = sycl::ext::oneapi::detail::get_wi_data(sg, mb);
#pragma unroll
    for (int i = 0; i < 16; ++i) wb[i] = v[i];
}

template <int TYPE> constexpr int qk() { return TYPE == kQ3K ? 256 : TYPE == kQ20 ? 64 : 128; }

/// Block `blk` of NM matrices (row `row` in each, planar, NB blocks a matrix) against the activations of K offset k0:
/// c[i][t] += block dot of matrix i with token row t.  The joint matrices are named objects (an array of them does
/// not lower: IGC "__spirv_AccessChain ... must be pointer to target extension type", and wrong rows where it built).
template <int TYPE, int NM>
inline void blk_x(sycl::sub_group sg, const uint8_t* const* m, size_t NB, size_t blk, int k0, const sycl::half* xt, int ln,
                  float (&c)[NM][8]) {
    Acc cb0, cb1;
    mx::joint_matrix_fill(sg, cb0, 0.0f);
    if constexpr (NM > 1) mx::joint_matrix_fill(sg, cb1, 0.0f);
    float d[NM];
    const h2 one((sycl::half) 1.0f, (sycl::half) 1.0f);
    // one product step: A at K offset k, B of matrix i from v
    auto step = [&](MatA& ma, int i, const sycl::half (&v)[16]) {
        MatB mb;
        put_b(sg, mb, v);
        if (i == 0) mx::joint_matrix_mad(sg, cb0, ma, mb, cb0);
        else if constexpr (NM > 1) mx::joint_matrix_mad(sg, cb1, ma, mb, cb1);
    };
    if constexpr (TYPE == kQ20) {   // 64 values: 16 bytes of 2-bit codes (unit u = word u: value 4a + b at byte b bits 2a)
        uint32_t w[NM][4];
#pragma unroll
        for (int i = 0; i < NM; ++i) {
            const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(m[i] + blk * 16);
            w[i][0] = v.x(); w[i][1] = v.y(); w[i][2] = v.z(); w[i][3] = v.w();
            d[i] = hf(*reinterpret_cast<const uint16_t*>(m[i] + 16 * NB + blk * 2));
        }
        const h2 off((sycl::half) 1025.0f, (sycl::half) 1025.0f);   // code - 1
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            MatA ma;
            load_a(sg, ma, xt, k0 + 16 * u, ln);
#pragma unroll
            for (int i = 0; i < NM; ++i) {
                sycl::half v[16];
#pragma unroll
                for (int a = 0; a < 4; ++a) bytes4((w[i][u] >> (2 * a)) & 0x03030303u, off, one, v + 4 * a);
                step(ma, i, v);
            }
        }
    } else if constexpr (TYPE == kPtq1) {   // 128 values: 16 + 8 + 2 bytes of base-3 trits, d in the last 2
        uint32_t W16[NM][4], W8[NM][2], W4[NM];
#pragma unroll
        for (int i = 0; i < NM; ++i) {
            const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(m[i] + blk * 16);
            const sycl::uint2 b = *reinterpret_cast<const sycl::uint2*>(m[i] + 16 * NB + blk * 8);
            const uint32_t p4 = *reinterpret_cast<const uint32_t*>(m[i] + 24 * NB + blk * 4);
            W16[i][0] = a.x(); W16[i][1] = a.y(); W16[i][2] = a.z(); W16[i][3] = a.w();
            W8[i][0] = b.x(); W8[i][1] = b.y();
            W4[i] = p4 & 0xFFFFu;
            d[i] = hf((uint16_t) (p4 >> 16));
        }
        const h2 off((sycl::half) 1025.0f, (sycl::half) 1025.0f);   // trit - 1
#pragma unroll
        for (int s = 0; s < 8; ++s) {   // values 16 s .. 16 s + 15
            MatA ma;
            load_a(sg, ma, xt, k0 + 16 * s, ln);
#pragma unroll
            for (int i = 0; i < NM; ++i) {
                sycl::half v[16];
                if (s < 5) {   // byte j of the 16, digit s
#pragma unroll
                    for (int w = 0; w < 4; ++w) bytes4(trits(W16[i][w], s), off, one, v + 4 * w);
                } else if (s < 7) {   // the 8, digits 2 (s - 5) and 2 (s - 5) + 1: values 0..7, 8..15
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int w = 0; w < 2; ++w) bytes4(trits(W8[i][w], 2 * (s - 5) + h), off, one, v + 8 * h + 4 * w);
                } else {   // the 8, digit 4 (values 0..7); the 2, digits 0..3 alternating bytes (values 8..15)
#pragma unroll
                    for (int w = 0; w < 2; ++w) bytes4(trits(W8[i][w], 4), off, one, v + 4 * w);
#pragma unroll
                    for (int dg = 0; dg < 4; ++dg) {
                        sycl::half o4[4];
                        bytes4(trits(W4[i], dg), off, one, o4);
                        v[8 + 2 * dg] = o4[0];
                        v[9 + 2 * dg] = o4[1];
                    }
                }
                step(ma, i, v);
            }
        }
    } else {   // Q3_K, 256 values: v = 128 h + 32 jj + l: q = q2 + 4 hbit - 4, scale 8 h + 2 jj + (l >= 16), d
        uint32_t sc[NM][3];
#pragma unroll
        for (int i = 0; i < NM; ++i) {
            const uint32_t* ps = reinterpret_cast<const uint32_t*>(m[i] + 96 * NB + blk * 12);
            sc[i][0] = ps[0]; sc[i][1] = ps[1]; sc[i][2] = ps[2];
            d[i] = hf(*reinterpret_cast<const uint16_t*>(m[i] + 108 * NB + blk * 2));
        }
        const h2 off((sycl::half) 1028.0f, (sycl::half) 1028.0f);   // (q + 4) - 4
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int s = 0; s < 2; ++s) {   // l0 = 16 s: qs bytes 32 h + l0 .., hm bytes l0 ..
                uint32_t qs[NM][4], hm[NM][4];
#pragma unroll
                for (int i = 0; i < NM; ++i) {
                    const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(m[i] + blk * 64 + 32 * h + 16 * s);
                    const sycl::uint4 b = *reinterpret_cast<const sycl::uint4*>(m[i] + 64 * NB + blk * 32 + 16 * s);
                    qs[i][0] = a.x(); qs[i][1] = a.y(); qs[i][2] = a.z(); qs[i][3] = a.w();
                    hm[i][0] = b.x(); hm[i][1] = b.y(); hm[i][2] = b.z(); hm[i][3] = b.w();
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    MatA ma;
                    load_a(sg, ma, xt, k0 + 128 * h + 32 * jj + 16 * s, ln);
                    const int is = 8 * h + 2 * jj + s;
#pragma unroll
                    for (int i = 0; i < NM; ++i) {
                        const uint32_t lw = (is % 8) < 4 ? sc[i][0] : sc[i][1];
                        const uint32_t lo = (lw >> (8 * (is % 4) + 4 * (is / 8))) & 0xFu;
                        const uint32_t hi = (sc[i][2] >> (8 * (is % 4) + 2 * (is / 4))) & 0x3u;
                        const sycl::half sm = (sycl::half) (float) ((int) (lo | (hi << 4)) - 32);
                        const h2 s2(sm, sm);
                        sycl::half v[16];
#pragma unroll
                        for (int w = 0; w < 4; ++w)
                            bytes4(((qs[i][w] >> (2 * jj)) & 0x03030303u) | (((hm[i][w] >> (4 * h + jj)) & 0x01010101u) << 2), off, s2, v + 4 * w);
                        step(ma, i, v);
                    }
                }
            }
    }
    {
        auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, cb0);
#pragma unroll
        for (int r = 0; r < 8; ++r) c[0][r] += (float) wc[r] * d[0];
    }
    if constexpr (NM > 1) {
        auto wc = sycl::ext::oneapi::detail::get_wi_data(sg, cb1);
#pragma unroll
        for (int r = 0; r < 8; ++r) c[NM - 1][r] += (float) wc[r] * d[NM - 1];
    }
}

/// the row's blocks [b0, b1) of NM matrices with n-value rows
template <int TYPE, int NM>
inline void row_x(sycl::sub_group sg, const uint8_t* const* m, int nrows, int n, int row, int b0, int b1, const sycl::half* xt,
                  int ln, float (&c)[NM][8]) {
    constexpr int Q = qk<TYPE>();
    const int nbr = n / Q;
    const size_t NB = (size_t) nrows * nbr;
    for (int b = b0; b < b1; ++b) blk_x<TYPE, NM>(sg, m, NB, (size_t) row * nbr + b, b * Q, xt, ln, c);
}

inline float pick8(const float (&v)[8], int t) {   // v[t] without a private-array index
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) r = i == t ? v[i] : r;
    return r;
}

template <int KS, int GRF>
sycl::event gate_up_x(sycl::queue& q, const XeJob* jobs, int nj, const sycl::half* xt, sycl::half* ht, int E, int F, float L) {
    const int nrg = F / 16;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * nrg * KS * SG), sycl::range<1>((size_t) KS * SG));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(KS > 1 ? (size_t) KS * 2 * 16 * 8 : 1), h);
        auto kern = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int ln = (int) sg.get_local_linear_id();
            const int ks = (int) sg.get_group_linear_id();
            const int gi = (int) it.get_group(0);
            const int j = gi / nrg, row = (gi % nrg) * 16 + ln;
            const XeJob& jb = jobs[j];
            const int nt = jb.nt;
            if (nt == 0) return;   // a null job: no H (the down pass skips it), the whole work-group leaves
            const uint8_t* mats[2] = {jb.blob, jb.blob + jb.up_off};
            float c[2][8] = {};
            const int type = jb.gu_type;
            const int Q = type == kQ3K ? 256 : type == kQ20 ? 64 : 128;
            const int nb = E / Q, b0 = nb * ks / KS, b1 = nb * (ks + 1) / KS;
            if (type == kQ3K) row_x<kQ3K, 2>(sg, mats, F, E, row, b0, b1, xt, ln, c);
            else if (type == kQ20) row_x<kQ20, 2>(sg, mats, F, E, row, b0, b1, xt, ln, c);
            else row_x<kPtq1, 2>(sg, mats, F, E, row, b0, b1, xt, ln, c);
            if constexpr (KS > 1) {
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int r = 0; r < 8; ++r) part[(((size_t) ks * 2 + i) * 16 + ln) * 8 + r] = c[i][r];
                it.barrier(sycl::access::fence_space::local_space);
                if (ks != 0) return;
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        float s = 0.0f;
#pragma unroll
                        for (int k2 = 0; k2 < KS; ++k2) s += part[(((size_t) k2 * 2 + i) * 16 + ln) * 8 + r];
                        c[i][r] = s;
                    }
            }
            // the job's tokens: H[j][row][s] = silu(min(g, L)) * clamp(u, -L, L) of token tok[s] (slots >= nt: 0)
            sycl::vec<sycl::half, 8> hv;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                float v = 0.0f;
                if (s < nt) {
                    const int t = jb.tok[s];
                    const float g = sycl::fmin(pick8(c[0], t), L);
                    const float u = sycl::fmin(sycl::fmax(pick8(c[1], t), -L), L);
                    v = g / (1.0f + sycl::exp(-g)) * u;
                }
                hv[s] = (sycl::half) v;
            }
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(ht + ((size_t) j * F + row) * 8) = hv;
        };
        h.parallel_for(nr, sxo::properties{sycl::ext::intel::experimental::grf_size<GRF>}, kern);
    });
}

template <int KS, int GRF>
sycl::event down_x(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const sycl::half* ht, float* OUT, int E, int F,
                   bool accumulate) {
    const int nrg = E / 16;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nrg * KS * SG), sycl::range<1>((size_t) KS * SG));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(KS > 1 ? (size_t) KS * 16 * 8 : 1), h);
        auto kern = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int ln = (int) sg.get_local_linear_id();
            const int ks = (int) sg.get_group_linear_id();
            const int row = (int) it.get_group(0) * 16 + ln;
            float o[8] = {};
            for (int j = 0; j < nj; ++j) {
                const XeJob& jb = jobs[j];
                const int nt = jb.nt;
                if (nt == 0) continue;
                const uint8_t* mats[1] = {jb.blob + jb.down_off};
                float c[1][8] = {};
                const int type = jb.d_type;
                const int Q = type == kQ3K ? 256 : type == kQ20 ? 64 : 128;
                const int nb = F / Q, b0 = nb * ks / KS, b1 = nb * (ks + 1) / KS;
                const sycl::half* hj = ht + (size_t) j * F * 8;
                if (type == kQ3K) row_x<kQ3K, 1>(sg, mats, E, F, row, b0, b1, hj, ln, c);
                else if (type == kQ20) row_x<kQ20, 1>(sg, mats, E, F, row, b0, b1, hj, ln, c);
                else row_x<kPtq1, 1>(sg, mats, E, F, row, b0, b1, hj, ln, c);
#pragma unroll
                for (int s = 0; s < 8; ++s)
                    if (s < nt) {
                        const int t = jb.tok[s];
                        const float v = jb.w[s] * c[0][s];
#pragma unroll
                        for (int i = 0; i < 8; ++i) o[i] += i == t ? v : 0.0f;
                    }
            }
            if constexpr (KS > 1) {
#pragma unroll
                for (int r = 0; r < 8; ++r) part[((size_t) ks * 16 + ln) * 8 + r] = o[r];
                it.barrier(sycl::access::fence_space::local_space);
                if (ks != 0) return;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    float s = 0.0f;
#pragma unroll
                    for (int k2 = 0; k2 < KS; ++k2) s += part[((size_t) k2 * 16 + ln) * 8 + r];
                    o[r] = s;
                }
            }
#pragma unroll
            for (int t = 0; t < 8; ++t)
                if (t < ntok) {
                    float* p = OUT + (size_t) t * E + row;
                    *p = accumulate ? *p + o[t] : o[t];
                }
        };
        h.parallel_for(nr, sxo::properties{sycl::ext::intel::experimental::grf_size<GRF>}, kern);
    });
}

}  // namespace

sycl::event xe_xt(sycl::queue& q, const float* x, int ntok, int n, sycl::half* xt) {
    return q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t k = id[0];
        sycl::vec<sycl::half, 8> v;
#pragma unroll
        for (int t = 0; t < 8; ++t) v[t] = t < ntok ? (sycl::half) x[(size_t) t * n + k] : (sycl::half) 0.0f;
        *reinterpret_cast<sycl::vec<sycl::half, 8>*>(xt + k * 8) = v;
    });
}

sycl::event xe_experts_x(sycl::queue& q, const XeJob* jobs, int nj, int ntok, const sycl::half* xt, sycl::half* ht,
                         float* OUT, int n_embd, int n_ff, float swiglu_limit, bool accumulate) {
    // K splits that divide every format's blocks a row: n_embd 5120 = 20 Q3_K / 40 PTQ1 / 80 Q2_0 blocks (4, 5, 10, 20),
    // n_ff 2304 = 9 / 18 / 36 (3, 9).  STRATA_XMX_KSG / _KSD / _GRF (bench knobs): the gate/up and down splits, the
    // register mode (128 / 256)
    auto env = [](const char* k, int d) { const char* e = std::getenv(k); return e ? std::atoi(e) : d; };
    static const int ksg = env("STRATA_XMX_KSG", 10), ksd = env("STRATA_XMX_KSD", 9), grf = env("STRATA_XMX_GRF", 256);
    const bool g2 = grf == 256;
    if (ksg == 4) g2 ? gate_up_x<4, 256>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit) : gate_up_x<4, 128>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit);
    else if (ksg == 20) g2 ? gate_up_x<20, 256>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit) : gate_up_x<20, 128>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit);
    else g2 ? gate_up_x<10, 256>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit) : gate_up_x<10, 128>(q, jobs, nj, xt, ht, n_embd, n_ff, swiglu_limit);
    if (ksd == 3) return g2 ? down_x<3, 256>(q, jobs, nj, ntok, ht, OUT, n_embd, n_ff, accumulate) : down_x<3, 128>(q, jobs, nj, ntok, ht, OUT, n_embd, n_ff, accumulate);
    return g2 ? down_x<9, 256>(q, jobs, nj, ntok, ht, OUT, n_embd, n_ff, accumulate) : down_x<9, 128>(q, jobs, nj, ntok, ht, OUT, n_embd, n_ff, accumulate);
}

}  // namespace strata::v41::xe
