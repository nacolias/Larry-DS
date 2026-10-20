// tools/v41/engine/xe/xe_planar.cpp - the PLANAR int8 expert kernels (decode / verify; see xe_kernels.hpp).
//
// Why: the file layout's blocks are 110 / 18 / 28 bytes, so a block's fields sit at 2-byte alignment and the F32 kernels
// load them a byte or a half-word at a time; with F32 activations every weight also costs a convert and an FMA per token.
// Both made the decode kernels instruction-bound at ~125 GB/s (8-token verify: 31 GB/s).  Here
//   * an expert's three matrices are stored as PLANES (xe_repack_blob, same size): each field of every block of the
//     matrix contiguous, so the kernels load 16 / 8 / 4 bytes at a time, aligned;
//   * activations are int8 with an f32 scale per 32 values and an int sum per 16 (the CPU / CUDA tiers' Q8 rounding,
//     at a finer group), and every 4 weights x 4 activations is one dp4a (IGC's builtin: ~7x the C++ spelling);
//   * gate/up quantises its SwiGLU output for down in the same kernel (a work-group owns 32 consecutive H rows).
//
// Planar layout of a matrix of R rows x n values, NB = R * n / QK blocks, block index = row * (n / QK) + b:
//   Q3_K   (QK 256): qs[64] x NB | hmask[32] x NB | scales[12] x NB | d x NB        (the file's fields, unchanged)
//   Q2_0   (QK  64): qs'[16] x NB | d x NB, qs' re-encoded per 16 values: value 4k + i of a 16-group is byte i,
//                    bits 2k, so (word >> 2k) & 0x03030303 is 4 consecutive codes
//   PTQ1_0 (QK 128): trits[0..15] x NB | trits[16..23] x NB | (trits[24..25], d) x NB
#include "xe_kernels.hpp"

#include <algorithm>

namespace strata::v41::xe {
namespace {

constexpr int SG = 16;

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_us(int c, int a, int b, bool sat) __attribute__((const));
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat) __attribute__((const));
#endif

/// c + sum_i a.u8[i] * b.s8[i]
inline int dp4a(uint32_t a, uint32_t b, int c) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_IB_dp4a_us(c, (int) a, (int) b, false);
#else
    int s = c;
    for (int i = 0; i < 4; ++i) s += (int) ((a >> (8 * i)) & 0xFFu) * (int) (int8_t) ((b >> (8 * i)) & 0xFFu);
    return s;
#endif
}

/// c + sum_i a.s8[i] * b.s8[i]
inline int dp4a_ss(uint32_t a, uint32_t b, int c) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_IB_dp4a_ss(c, (int) a, (int) b, false);
#else
    int s = c;
    for (int i = 0; i < 4; ++i) s += (int) (int8_t) ((a >> (8 * i)) & 0xFFu) * (int) (int8_t) ((b >> (8 * i)) & 0xFFu);
    return s;
#endif
}

/// bytes v (0..2^7) -> v - off as signed bytes, no borrow between bytes (off < 128)
inline uint32_t sbytes(uint32_t v, uint32_t off) {
    const uint32_t k = (128u - off) * 0x01010101u;
    return (v + k) ^ 0x80808080u;
}

inline float h2f(uint16_t u) { return (float) sycl::bit_cast<sycl::half>(u); }

/// int8 activation rows (n values each): a job's tokens are rows[t] of it.
struct XSet {
    const int8_t* q;
    const float* d;
    const int32_t* s;
    int n;
    const int8_t* qrow(int r) const { return q + (size_t) r * n; }
    const float* drow(int r) const { return d + (size_t) r * (n / 32); }
    const int32_t* srow(int r) const { return s + (size_t) r * (n / 16); }
};

// ---- block dots: block `blk` of NR matrices (same block index), NT activation rows; acc[r][t] += row r . x_t ----

// Q3_K: a block is split over 4 lanes, chunk ci = 16 bytes of qs (h = ci / 2, s = ci % 2: l = 16s .. 16s + 15), the
// values 128h + 32jj + l for jj < 4, one scale each (index 8h + 2jj + s).  q = q2 + 4 hbit - 4, and the masked (not
// shifted) fields go straight into dp4a: sum (q2 << 2jj) x and sum (hbit << (4h + jj)) x are exact multiples.
template <int NT, int NR>
inline void blk_q3k(const uint8_t* const* m, size_t NB, size_t blk, int ci, const XSet& xs, const int* rows, int b, int nt,
                    float (&acc)[NR][NT]) {
    const int h = ci >> 1, s = ci & 1;
    uint32_t qs[NR][4], hm[NR][4];
    float fsc[NR][4];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(m[r] + blk * 64 + ci * 16);
        qs[r][0] = v.x(); qs[r][1] = v.y(); qs[r][2] = v.z(); qs[r][3] = v.w();
        const sycl::uint4 u = *reinterpret_cast<const sycl::uint4*>(m[r] + 64 * NB + blk * 32 + s * 16);
        hm[r][0] = u.x(); hm[r][1] = u.y(); hm[r][2] = u.z(); hm[r][3] = u.w();
        const uint32_t* ps = reinterpret_cast<const uint32_t*>(m[r] + 96 * NB + blk * 12);
        const uint32_t a0 = ps[0], a1 = ps[1], tt = ps[2];
        const float d = h2f(*reinterpret_cast<const uint16_t*>(m[r] + 108 * NB + blk * 2));
        // scale 8h + 2jj + s: low nibble h of byte 2jj + s, high bits 2h + jj / 2 (pairs) of byte 8 + (2jj + s) % 4
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const uint32_t lw = jj < 2 ? a0 : a1;
            const uint32_t lo = (lw >> (16 * (jj & 1) + 8 * s + 4 * h)) & 0xFu;
            const uint32_t hi = (tt >> (16 * (jj & 1) + 8 * s + 2 * (2 * h + jj / 2))) & 0x3u;
            fsc[r][jj] = d * (float) ((int) (lo | (hi << 4)) - 32);
        }
    }
    if constexpr (NT == 1) {
        // one token: the masked fields straight into dp4a, the -4 offset as a dp4a of the activations
        const int8_t* xq = xs.qrow(rows[0]) + (size_t) b * 256 + 128 * h + 16 * s;
        const sycl::float4 dx4 = *reinterpret_cast<const sycl::float4*>(xs.drow(rows[0]) + (size_t) b * 8 + 4 * h);
        const float dxs[4] = {dx4.x(), dx4.y(), dx4.z(), dx4.w()};
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const uint32_t Mq = 0x03030303u << (2 * jj), Mh = 0x01010101u << (4 * h + jj);
            const sycl::uint4 xv = *reinterpret_cast<const sycl::uint4*>(xq + 32 * jj);
            const uint32_t xw[4] = {xv.x(), xv.y(), xv.z(), xv.w()};
            int s4x = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) s4x = dp4a(0x04040404u, xw[w], s4x);
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                int aq = 0, ah = 0;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    aq = dp4a(qs[r][w] & Mq, xw[w], aq);
                    ah = dp4a(hm[r][w] & Mh, xw[w], ah);
                }
                const int S = (aq >> (2 * jj)) + ((ah >> (4 * h + jj)) << 2) - s4x;
                acc[r][0] += fsc[r][jj] * dxs[jj] * (float) S;
            }
        }
    } else {
        // several tokens: the signed values (q2 + 4 hbit - 4) once, then one dp4a per 4 values per token
        uint32_t qv[NR][4][4];
#pragma unroll
        for (int r = 0; r < NR; ++r)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
#pragma unroll
                for (int w = 0; w < 4; ++w)
                    qv[r][jj][w] = sbytes(((qs[r][w] >> (2 * jj)) & 0x03030303u) | (((hm[r][w] >> (4 * h + jj)) & 0x01010101u) << 2), 4);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < nt) {
                const int8_t* xq = xs.qrow(rows[t]) + (size_t) b * 256 + 128 * h + 16 * s;
                const sycl::float4 dx4 = *reinterpret_cast<const sycl::float4*>(xs.drow(rows[t]) + (size_t) b * 8 + 4 * h);
                const float dxs[4] = {dx4.x(), dx4.y(), dx4.z(), dx4.w()};
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const sycl::uint4 xv = *reinterpret_cast<const sycl::uint4*>(xq + 32 * jj);
                    const uint32_t xw[4] = {xv.x(), xv.y(), xv.z(), xv.w()};
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        int a = 0;
#pragma unroll
                        for (int w = 0; w < 4; ++w) a = dp4a_ss(qv[r][jj][w], xw[w], a);
                        acc[r][t] += fsc[r][jj] * dxs[jj] * (float) a;
                    }
                }
            }
        }
    }
}

template <int NT, int NR>
inline void blk_q20(const uint8_t* const* m, size_t NB, size_t blk, const XSet& xs, const int* rows, int b, int nt, float (&acc)[NR][NT]) {
    uint32_t w[NR][4];
    float dd[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(m[r] + blk * 16);
        w[r][0] = v.x(); w[r][1] = v.y(); w[r][2] = v.z(); w[r][3] = v.w();
        dd[r] = h2f(*reinterpret_cast<const uint16_t*>(m[r] + 16 * NB + blk * 2));
    }
    uint32_t cw[NR][16];   // cw[u * 4 + k]: the signed values (code - 1) 16u + 4k .. 16u + 4k + 3
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int k = 0; k < 4; ++k) cw[r][4 * u + k] = sbytes((w[r][u] >> (2 * k)) & 0x03030303u, 1);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t < nt) {
            const sycl::uint4* xp = reinterpret_cast<const sycl::uint4*>(xs.qrow(rows[t]) + (size_t) b * 64);
            uint32_t xw[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const sycl::uint4 v = xp[i];
                xw[4 * i] = v.x(); xw[4 * i + 1] = v.y(); xw[4 * i + 2] = v.z(); xw[4 * i + 3] = v.w();
            }
            const sycl::float2 xd = *reinterpret_cast<const sycl::float2*>(xs.drow(rows[t]) + (size_t) b * 2);
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) s0 = dp4a_ss(cw[r][i], xw[i], s0);
#pragma unroll
                for (int i = 8; i < 16; ++i) s1 = dp4a_ss(cw[r][i], xw[i], s1);
                acc[r][t] += dd[r] * (xd.x() * (float) s0 + xd.y() * (float) s1);
            }
        }
    }
}

// PTQ1_0: trit d of byte b is t_d = F_d - 3 F_(d-1), F_d = floor(b * 3^(d+1) / 256) (F_-1 = 0); the F of 4 bytes
// come out of two 16-bit-lane multiplies as one word, so sum t_d x_d = dp4a(F_d, X_d) - 3 dp4a(F_(d-1), X_d), exact.
inline uint32_t fword(uint32_t lo, uint32_t hi, uint32_t p) {
    return (((lo * p) >> 8) & 0x00FF00FFu) | ((hi * p) & 0xFF00FF00u);
}

template <int NT, int NR>
inline void blk_ptq1(const uint8_t* const* m, size_t NB, size_t blk, const XSet& xs, const int* rows, int b, int nt, float (&acc)[NR][NT]) {
    uint32_t W[NR][7];
    float dd[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(m[r] + blk * 16);
        const sycl::uint2 c = *reinterpret_cast<const sycl::uint2*>(m[r] + 16 * NB + blk * 8);
        const uint32_t p4 = *reinterpret_cast<const uint32_t*>(m[r] + 24 * NB + blk * 4);
        W[r][0] = a.x(); W[r][1] = a.y(); W[r][2] = a.z(); W[r][3] = a.w();
        W[r][4] = c.x(); W[r][5] = c.y(); W[r][6] = p4 & 0xFFFFu;
        dd[r] = h2f((uint16_t) (p4 >> 16));
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t < nt) {
            const sycl::uint4* xp = reinterpret_cast<const sycl::uint4*>(xs.qrow(rows[t]) + (size_t) b * 128);
            uint32_t xw[32];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const sycl::uint4 v = xp[i];
                xw[4 * i] = v.x(); xw[4 * i + 1] = v.y(); xw[4 * i + 2] = v.z(); xw[4 * i + 3] = v.w();
            }
            const sycl::float4 xd4 = *reinterpret_cast<const sycl::float4*>(xs.drow(rows[t]) + (size_t) b * 4);
            const float xd[4] = {xd4.x(), xd4.y(), xd4.z(), xd4.w()};
            const sycl::int4 xs0 = *reinterpret_cast<const sycl::int4*>(xs.srow(rows[t]) + (size_t) b * 8);
            const sycl::int4 xs1 = *reinterpret_cast<const sycl::int4*>(xs.srow(rows[t]) + (size_t) b * 8 + 4);
            const int xsum[8] = {xs0.x(), xs0.y(), xs0.z(), xs0.w(), xs1.x(), xs1.y(), xs1.z(), xs1.w()};
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                int A[4] = {0, 0, 0, 0}, B[4] = {0, 0, 0, 0};   // per activation group: sum F_d X_d, sum F_(d-1) X_d
                // bytes 0..15: byte 4q + i, digit d = value 16d + 4q + i (group d / 2)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t lo = W[r][q] & 0x00FF00FFu, hi = (W[r][q] >> 8) & 0x00FF00FFu;
                    uint32_t p = 3, Fp = 0;
#pragma unroll
                    for (int d = 0; d < 5; ++d) {
                        const uint32_t F = fword(lo, hi, p);
                        const int g = d / 2;
                        const uint32_t X = xw[4 * d + q];
                        A[g] = dp4a(F, X, A[g]);
                        if (d > 0) B[g] = dp4a(Fp, X, B[g]);
                        Fp = F;
                        p *= 3;
                    }
                }
                // bytes 16..23: byte 16 + 4q + i, digit d = value 80 + 8d + 4q + i (group 2 for d < 2, else 3)
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t lo = W[r][4 + q] & 0x00FF00FFu, hi = (W[r][4 + q] >> 8) & 0x00FF00FFu;
                    uint32_t p = 3, Fp = 0;
#pragma unroll
                    for (int d = 0; d < 5; ++d) {
                        const uint32_t F = fword(lo, hi, p);
                        const int g = d < 2 ? 2 : 3;
                        const uint32_t X = xw[20 + 2 * d + q];
                        A[g] = dp4a(F, X, A[g]);
                        if (d > 0) B[g] = dp4a(Fp, X, B[g]);
                        Fp = F;
                        p *= 3;
                    }
                }
                // bytes 24, 25: digit d (< 4) = values 120 + 2d, 121 + 2d (group 3): word 30 + d / 2, bytes 0-1 or 2-3
                {
                    const uint32_t lo = W[r][6] & 0x00FF00FFu, hi = (W[r][6] >> 8) & 0x00FF00FFu;
                    uint32_t p = 3, Fp = 0;
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const uint32_t F = fword(lo, hi, p) & 0xFFFFu;
                        const int sh = (d & 1) ? 16 : 0;
                        const uint32_t X = xw[30 + d / 2];
                        A[3] = dp4a(F << sh, X, A[3]);
                        if (d > 0) B[3] = dp4a(Fp << sh, X, B[3]);
                        Fp = F;
                        p *= 3;
                    }
                }
                float s = 0.f;
#pragma unroll
                for (int g = 0; g < 4; ++g) s += xd[g] * (float) (A[g] - 3 * B[g] - xsum[2 * g] - xsum[2 * g + 1]);   // trit = value + 1
                acc[r][t] += dd[r] * s;
            }
        }
    }
}

template <int TYPE, int NT, int NR>
inline void blk_dot(const uint8_t* const* m, size_t NB, size_t blk, int ci, const XSet& xs, const int* rows, int b, int nt,
                    float (&acc)[NR][NT]) {
    if constexpr (TYPE == kQ3K) blk_q3k<NT, NR>(m, NB, blk, ci, xs, rows, b, nt, acc);
    else if constexpr (TYPE == kQ20) blk_q20<NT, NR>(m, NB, blk, xs, rows, b, nt, acc);
    else blk_ptq1<NT, NR>(m, NB, blk, xs, rows, b, nt, acc);
}

inline int qk_of(int type) { return type == kQ3K ? 256 : type == kQ20 ? 64 : 128; }
template <int TYPE>
constexpr int qk_c() { return TYPE == kQ3K ? 256 : TYPE == kQ20 ? 64 : 128; }

// ---- gate/up + SwiGLU + int8 quantisation of H.  A work-group (8 sub-groups) owns 32 consecutive rows of one job;
// a sub-group 4 rows, 4 lanes per row (every 4th block); each lane does the gate AND the up row (shared x loads).
template <int TYPE, int NT>
inline void gu_rows(const XeJob& jb, const XSet& xs, const int* rows, int row, int bl, int n_ff, int nt, float (&acc)[2][NT]) {
    const uint8_t* mats[2] = {jb.blob, jb.blob + jb.up_off};
    const int nbr = xs.n / qk_c<TYPE>();
    const size_t NB = (size_t) n_ff * nbr;
    if constexpr (TYPE == kQ3K) {   // 4 lanes per block (chunk bl): a row's 4 lanes read whole cache lines
#pragma unroll 2
        for (int b = 0; b < nbr; ++b) blk_dot<TYPE, NT, 2>(mats, NB, (size_t) row * nbr + b, bl, xs, rows, b, nt, acc);
    } else {                        // a block per lane: 4 consecutive blocks per row per load
#pragma unroll 2
        for (int b = bl; b < nbr; b += 4) blk_dot<TYPE, NT, 2>(mats, NB, (size_t) row * nbr + b, 0, xs, rows, b, nt, acc);
    }
}

template <int NT>
sycl::event gate_up_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 X, XeQ8 Hq, int n_embd, int n_ff, float L) {
    const int rb = n_ff / 32;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * rb * 128), sycl::range<1>(128));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(NT * 32), h);
        h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int wg = (int) it.get_group(0);
            const int j = wg / rb, r0 = (wg % rb) * 32;
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int rg = lane >> 2, bl = lane & 3;
            const int row = r0 + s * 4 + rg;
            const XeJob& jb = jobs[j];
            const int nt = jb.nt;
            const XSet xs{X.q, X.d, X.s, n_embd};
            int rows[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) rows[t] = jb.tok[t < nt ? t : 0];
            float acc[2][NT];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
            const int type = jb.gu_type;
            if (type == kQ3K) gu_rows<kQ3K, NT>(jb, xs, rows, row, bl, n_ff, nt, acc);
            else if (type == kQ20) gu_rows<kQ20, NT>(jb, xs, rows, row, bl, n_ff, nt, acc);
            else gu_rows<kPtq1, NT>(jb, xs, rows, row, bl, n_ff, nt, acc);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    float v = acc[i][t];
                    v += sycl::permute_group_by_xor(sg, v, 1);
                    v += sycl::permute_group_by_xor(sg, v, 2);
                    acc[i][t] = v;
                }
            if (bl == 0) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < nt) {
                        const float g = sycl::fmin(acc[0][t], L);
                        const float u = sycl::fmax(-L, sycl::fmin(acc[1][t], L));
                        sh[t * 32 + s * 4 + rg] = g / (1.0f + sycl::exp(-g)) * u;
                    }
                }
            }
            sycl::group_barrier(it.get_group());
            // sub-group s quantises token slot s's 32 values: lane = values 2 lane, 2 lane + 1
            if (s < nt) {
                const float h0 = sh[s * 32 + 2 * lane], h1 = sh[s * 32 + 2 * lane + 1];
                const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(h0), sycl::fabs(h1)), sycl::maximum<float>());
                const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
                const int q0 = (int) sycl::rint(h0 * id), q1 = (int) sycl::rint(h1 * id);
                const int ps = q0 + q1;
                const int slo = sycl::reduce_over_group(sg, lane < 8 ? ps : 0, sycl::plus<int>());
                const int shi = sycl::reduce_over_group(sg, lane < 8 ? 0 : ps, sycl::plus<int>());
                const size_t hrow = (size_t) j * kMaxJobTok + s;
                *reinterpret_cast<uint16_t*>(Hq.q + hrow * n_ff + r0 + 2 * lane) = (uint16_t) ((q0 & 0xFF) | ((q1 & 0xFF) << 8));
                if (lane == 0) {
                    Hq.d[hrow * (n_ff / 32) + r0 / 32] = amax / 127.0f;
                    Hq.s[hrow * (n_ff / 16) + r0 / 16] = slo;
                    Hq.s[hrow * (n_ff / 16) + r0 / 16 + 1] = shi;
                }
            }
        });
    });
}

// ---- down + weighted sum over the jobs.  A sub-group owns 4 output rows with 4 lanes each (the gate/up kernel's lane
// layout: Q3_K a block's 4 chunks, the others every 4th block), loops over the jobs and writes its rows.
constexpr int kDownSgPerWg = 8;

template <int TYPE, int NT, int NTOK>
inline void down_job(const XeJob& jb, const XSet& hs, const int* rows, int row, int bl, int n_embd, int nt,
                     const sycl::sub_group& sg, float (&o)[NTOK]) {
    const uint8_t* mm[1] = {jb.blob + jb.down_off};
    const int nbr = hs.n / qk_c<TYPE>();
    const size_t NB = (size_t) n_embd * nbr;
    float a[1][NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) a[0][t] = 0.f;
    if constexpr (TYPE == kQ3K) {
#pragma unroll 3
        for (int b = 0; b < nbr; ++b) blk_dot<TYPE, NT, 1>(mm, NB, (size_t) row * nbr + b, bl, hs, rows, b, nt, a);
    } else {
#pragma unroll 3
        for (int b = bl; b < nbr; b += 4) blk_dot<TYPE, NT, 1>(mm, NB, (size_t) row * nbr + b, 0, hs, rows, b, nt, a);
    }
#pragma unroll
    for (int s = 0; s < NT; ++s) {
        float v = a[0][s];
        v += sycl::permute_group_by_xor(sg, v, 1);
        v += sycl::permute_group_by_xor(sg, v, 2);
        if (s < nt) {
            v *= jb.w[s];
            const int tk = jb.tok[s];
#pragma unroll
            for (int t = 0; t < NTOK; ++t) if (t == tk) o[t] += v;
        }
    }
}

template <int NT, int NTOK>   // NT: tokens per job (bound), NTOK: tokens of the call (bound)
sycl::event down_q8(sycl::queue& q, const XeJob* jobs, int nj, int ntok, XeQ8 Hq, float* OUT, int n_embd, int n_ff,
                    bool accumulate) {
    const int rows_wg = kDownSgPerWg * 4;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) (n_embd / rows_wg) * kDownSgPerWg * SG), sycl::range<1>(kDownSgPerWg * SG));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int rg = lane >> 2, bl = lane & 3;
        const int row = (int) it.get_group(0) * rows_wg + (int) sg.get_group_linear_id() * 4 + rg;
        const XSet hs{Hq.q, Hq.d, Hq.s, n_ff};
        float o[NTOK];
#pragma unroll
        for (int t = 0; t < NTOK; ++t) o[t] = 0.f;
        for (int j = 0; j < nj; ++j) {
            const XeJob& jb = jobs[j];
            const int nt = jb.nt;
            int rows[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) rows[t] = j * kMaxJobTok + (t < nt ? t : 0);
            const int type = jb.d_type;
            if (type == kQ3K) down_job<kQ3K, NT, NTOK>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
            else if (type == kPtq1) down_job<kPtq1, NT, NTOK>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
            else down_job<kQ20, NT, NTOK>(jb, hs, rows, row, bl, n_embd, nt, sg, o);
        }
        if (bl == 0) {
#pragma unroll
            for (int t = 0; t < NTOK; ++t)
                if (t < ntok) {
                    float* p = OUT + (size_t) t * n_embd + row;
                    *p = accumulate ? *p + o[t] : o[t];
                }
        }
    });
}

// ---- repack: one work-item per block of a matrix
sycl::event repack_mat(sycl::queue& q, int type, const uint8_t* src, uint8_t* dst, size_t NB) {
    return q.parallel_for(sycl::range<1>(NB), [=](sycl::id<1> id) {
        const size_t i = id[0];
        if (type == kQ3K) {
            const uint8_t* s = src + i * 110;
            for (int k = 0; k < 64; ++k) dst[i * 64 + k] = s[32 + k];
            for (int k = 0; k < 32; ++k) dst[64 * NB + i * 32 + k] = s[k];
            for (int k = 0; k < 12; ++k) dst[96 * NB + i * 12 + k] = s[96 + k];
            dst[108 * NB + i * 2] = s[108];
            dst[108 * NB + i * 2 + 1] = s[109];
        } else if (type == kQ20) {
            const uint8_t* s = src + i * 18;
            for (int u = 0; u < 4; ++u) {   // 16-value unit u: file value j' at byte j'/4 bits 2(j'%4) -> byte j'%4 bits 2(j'/4)
                uint8_t o[4] = {0, 0, 0, 0};
                for (int jp = 0; jp < 16; ++jp) {
                    const int code = (s[2 + 4 * u + jp / 4] >> (2 * (jp % 4))) & 3;
                    o[jp % 4] |= (uint8_t) (code << (2 * (jp / 4)));
                }
                for (int k = 0; k < 4; ++k) dst[i * 16 + 4 * u + k] = o[k];
            }
            dst[16 * NB + i * 2] = s[0];
            dst[16 * NB + i * 2 + 1] = s[1];
        } else {
            const uint8_t* s = src + i * 28;
            for (int k = 0; k < 16; ++k) dst[i * 16 + k] = s[k];
            for (int k = 0; k < 8; ++k) dst[16 * NB + i * 8 + k] = s[16 + k];
            for (int k = 0; k < 4; ++k) dst[24 * NB + i * 4 + k] = s[24 + k];
        }
    });
}

// ---- down_y: Y[j][s][i] = down row i of job j . H[j][s] (the prompt path's mat-vec jobs: any number of jobs).
// A sub-group 4 rows x 4 lanes as above; a work-group 8 sub-groups = 32 rows of one job.
template <int NT>
sycl::event down_y_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 Hq, float* Y, int n_embd, int n_ff) {
    const int rb = n_embd / 32;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) nj * rb * 128), sycl::range<1>(128));
    return q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int wg = (int) it.get_group(0);
        const int j = wg / rb;
        const int lane = (int) sg.get_local_linear_id();
        const int rg = lane >> 2, bl = lane & 3;
        const int row = (wg % rb) * 32 + (int) sg.get_group_linear_id() * 4 + rg;
        const XSet hs{Hq.q, Hq.d, Hq.s, n_ff};
        const XeJob& jb = jobs[j];
        const int nt = jb.nt;
        int rows[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) rows[t] = j * kMaxJobTok + (t < nt ? t : 0);
        const uint8_t* mm[1] = {jb.blob + jb.down_off};
        float a[1][NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) a[0][t] = 0.f;
        const int type = jb.d_type;
        const int nbr = n_ff / qk_of(type);
        const size_t NB = (size_t) n_embd * nbr;
        if (type == kQ3K) {
            for (int b = 0; b < nbr; ++b) blk_dot<kQ3K, NT, 1>(mm, NB, (size_t) row * nbr + b, bl, hs, rows, b, nt, a);
        } else if (type == kQ20) {
            for (int b = bl; b < nbr; b += 4) blk_dot<kQ20, NT, 1>(mm, NB, (size_t) row * nbr + b, 0, hs, rows, b, nt, a);
        } else {
            for (int b = bl; b < nbr; b += 4) blk_dot<kPtq1, NT, 1>(mm, NB, (size_t) row * nbr + b, 0, hs, rows, b, nt, a);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = a[0][t];
            v += sycl::permute_group_by_xor(sg, v, 1);
            v += sycl::permute_group_by_xor(sg, v, 2);
            if (bl == 0 && t < nt) Y[((size_t) j * kMaxJobTok + t) * n_embd + row] = v;
        }
    });
}

}  // namespace

sycl::event xe_repack_blob(sycl::queue& q, const uint8_t* src, uint8_t* dst, int gu_type, int d_type, size_t up_off,
                           size_t down_off, int n_embd, int n_ff) {
    const size_t nbg = (size_t) n_ff * (n_embd / qk_of(gu_type));
    const size_t nbd = (size_t) n_embd * (n_ff / qk_of(d_type));
    repack_mat(q, gu_type, src, dst, nbg);
    repack_mat(q, gu_type, src + up_off, dst + up_off, nbg);
    return repack_mat(q, d_type, src + down_off, dst + down_off, nbd);
}

sycl::event xe_quant_rows(sycl::queue& q, const float* x, int nrows, int n, XeQ8 out) {
    const int ng = n / 32;
    return q.parallel_for(sycl::range<1>((size_t) nrows * ng), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / ng), g = (int) (id[0] % ng);
        const float* p = x + (size_t) r * n + (size_t) g * 32;
        float v[32];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const sycl::float4 f = reinterpret_cast<const sycl::float4*>(p)[i];
            v[4 * i] = f.x(); v[4 * i + 1] = f.y(); v[4 * i + 2] = f.z(); v[4 * i + 3] = f.w();
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
        const float id_ = amax > 0.f ? 127.0f / amax : 0.f;
        uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int qv = (int) sycl::rint(v[i] * id_);
            w[i / 4] |= (uint32_t) (qv & 0xFF) << (8 * (i % 4));
            if (i < 16) s0 += qv; else s1 += qv;
        }
        sycl::uint4* o = reinterpret_cast<sycl::uint4*>(out.q + (size_t) r * n + (size_t) g * 32);
        o[0] = sycl::uint4(w[0], w[1], w[2], w[3]);
        o[1] = sycl::uint4(w[4], w[5], w[6], w[7]);
        out.d[(size_t) r * ng + g] = amax / 127.0f;
        out.s[(size_t) r * 2 * ng + 2 * g] = s0;
        out.s[(size_t) r * 2 * ng + 2 * g + 1] = s1;
    });
}

sycl::event xe_experts_y_q8(sycl::queue& q, const XeJob* jobs, int nj, XeQ8 X, XeQ8 H, float* Y, int n_embd, int n_ff,
                            float swiglu_limit, int max_nt) {
    if (max_nt <= 1) {
        gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
        return down_y_q8<1>(q, jobs, nj, H, Y, n_embd, n_ff);
    }
    if (max_nt <= 2) {
        gate_up_q8<2>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
        return down_y_q8<2>(q, jobs, nj, H, Y, n_embd, n_ff);
    }
    if (max_nt <= 4) {
        gate_up_q8<4>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
        return down_y_q8<4>(q, jobs, nj, H, Y, n_embd, n_ff);
    }
    gate_up_q8<kMaxJobTok>(q, jobs, nj, X, H, n_embd, n_ff, swiglu_limit);
    return down_y_q8<kMaxJobTok>(q, jobs, nj, H, Y, n_embd, n_ff);
}

// one work-item per 8 consecutive output values (as xe_dequant_rows_f16), from the planar layout
sycl::event xe_dequant_planar_f16(sycl::queue& q, int type, const uint8_t* mat, int nrows, int n, sycl::half* y) {
    using h8 = sycl::vec<sycl::half, 8>;
    const int per_row = n / 8;
    const size_t total = (size_t) nrows * per_row;
    const size_t NB = (size_t) nrows * (n / qk_of(type));
    return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int r = (int) (id[0] / per_row);
        const int v0 = (int) (id[0] % per_row) * 8;
        h8 o;
        if (type == kQ20) {
            const size_t blk = (size_t) r * (n / 64) + v0 / 64;
            const float d = h2f(*reinterpret_cast<const uint16_t*>(mat + 16 * NB + blk * 2));
            const int u = (v0 % 64) / 16, half = (v0 % 16) / 8;
            const uint32_t w = *reinterpret_cast<const uint32_t*>(mat + blk * 16 + 4 * u);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = (8 * half + j) / 4, i = j % 4;
                o[j] = (sycl::half) ((float) ((int) ((w >> (8 * i + 2 * k)) & 3u) - 1) * d);
            }
        } else if (type == kPtq1) {
            const size_t blk = (size_t) r * (n / 128) + v0 / 128;
            const uint8_t* p16 = mat + blk * 16;
            const uint8_t* p8 = mat + 16 * NB + blk * 8;
            const uint8_t* p4 = mat + 24 * NB + blk * 4;
            const float d = h2f((uint16_t) (p4[2] | (p4[3] << 8)));
            const int j0 = v0 % 128;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = j0 + i;
                int byte, dig;
                if (j < 80) { byte = j % 16; dig = j / 16; }
                else if (j < 120) { byte = 16 + (j - 80) % 8; dig = (j - 80) / 8; }
                else { byte = 24 + (j - 120) % 2; dig = (j - 120) / 2; }
                const uint32_t bv = byte < 16 ? p16[byte] : byte < 24 ? p8[byte - 16] : p4[byte - 24];
                const uint32_t p = dig == 0 ? 1u : dig == 1 ? 3u : dig == 2 ? 9u : dig == 3 ? 27u : 81u;
                o[i] = (sycl::half) ((float) ((int) ((((bv * p) & 0xFFu) * 3u) >> 8) - 1) * d);
            }
        } else {   // Q3_K: 8 values of one 32-group (h, jj), one half (one scale)
            const size_t blk = (size_t) r * (n / 256) + v0 / 256;
            const int j256 = v0 % 256;
            const int h = j256 / 128, jj = (j256 % 128) / 32, l0 = j256 % 32;
            const uint8_t* qs = mat + blk * 64;
            const uint8_t* hm = mat + 64 * NB + blk * 32;
            const uint8_t* sc = mat + 96 * NB + blk * 12;
            const int is = h * 8 + 2 * jj + (l0 >= 16 ? 1 : 0);
            const uint32_t lo4 = (sc[is % 8] >> (4 * (is / 8))) & 0x0fu;
            const uint32_t hi2 = (sc[8 + is % 4] >> (2 * (is / 4))) & 0x03u;
            const float dl = h2f(*reinterpret_cast<const uint16_t*>(mat + 108 * NB + blk * 2)) * (float) ((int) (lo4 | (hi2 << 4)) - 32);
            const int mbit = h * 4 + jj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int l = l0 + i;
                const int qv = (int) ((qs[h * 32 + l] >> (2 * jj)) & 3u) - (((hm[l] >> mbit) & 1u) ? 0 : 4);
                o[i] = (sycl::half) (dl * (float) qv);
            }
        }
        *reinterpret_cast<h8*>(y + (size_t) r * n + v0) = o;
    });
}

sycl::event xe_experts_q8(sycl::queue& q, const XeJob* jobs, int nj, int ntok, XeQ8 X, XeQ8 H, float* OUT, int n_embd,
                          int n_ff, float swiglu_limit, bool accumulate, int max_nt) {
    if (max_nt <= 0) max_nt = ntok;   // the most tokens any job has: the kernels' register footprint follows it
    const float Lm = swiglu_limit;
    if (ntok <= 1) {
        gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<1, 1>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    if (max_nt <= 1) {
        gate_up_q8<1>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<1, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    if (max_nt <= 2) {
        gate_up_q8<2>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<2, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    if (max_nt <= 4) {
        gate_up_q8<4>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
        return down_q8<4, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
    }
    gate_up_q8<kMaxJobTok>(q, jobs, nj, X, H, n_embd, n_ff, Lm);
    return down_q8<kMaxJobTok, kMaxJobTok>(q, jobs, nj, ntok, H, OUT, n_embd, n_ff, accumulate);
}

}  // namespace strata::v41::xe
