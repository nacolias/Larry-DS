// li_probe.cpp - larry: the prompt lightning indexer at the service's span shape (512 queries x 32 heads x 128 dims
// against cap keys; cap = the next power of two of the span's visible rows, the mask hides the rest).
//   ref   li_gemm as ggml-sycl runs it: Q, K -> fp16, one oneMKL GEMM writes S [B*H x cap] f32 in 256 MiB chunks, a
//         second kernel reads S back for the relu-weighted head sum (+ mask)
//   s16   the same with S in fp16 (GGML_SYCL_LI_S16)
//   fN    fused XMX kernels: S stays in registers.  A sub-group = one query (its 32 heads = 4 row tiles of 8) x 16*NT
//         keys a step, K = 128 in 8 steps; an accumulator lane holds its key's column (element i = row i, coord_probe),
//         so the head sum is lane-local.  Key steps the query cannot see (mask -inf for all of them) are skipped.
//         f2 / f4: NT 2 / 4, the query's A tiles re-loaded every step (L1); f2q: NT 2, the 32 A tiles kept in registers.
//         B (keys^T) col_major from the fp16 keys (LI_PACKED=0) or ext_intel_packed from a repacked copy (=1).
//   icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" li_probe.cpp \
//        -DMKL_ILP64 -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -ltbb -o li_probe
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./li_probe [cap:visible ...]
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <oneapi/mkl/blas.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifndef LI_PACKED
#define LI_PACKED 1
#endif

namespace mkl = oneapi::mkl;
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int64_t E = 128, H = 32;

using jA = mx::joint_matrix<sycl::sub_group, half, mx::use::a, 8, 16, mx::layout::row_major>;
#if LI_PACKED
using jB = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::ext_intel_packed>;
#else
using jB = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::col_major>;
#endif
using jC = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;

template <typename T> static inline auto gp(T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- the reference: li_gemm (lightning-indexer.cpp), S f32 or fp16
static void li_ref(sycl::queue & q, const float * qf, const bf16 * k, const float * w, const half * m, float * od,
                   int64_t B, int64_t NKV, half * qh, half * kh, void * Sbuf, bool s16) {
    q.parallel_for(sycl::range<1>((size_t) (B * H * E)), [=](sycl::id<1> i) { qh[i] = (half) qf[i]; });
    q.parallel_for(sycl::range<1>((size_t) (NKV * E)), [=](sycl::id<1> i) { kh[i] = (half) (float) k[i]; });
    const int64_t MR = B * H;
    const size_t es = s16 ? sizeof(half) : sizeof(float);
    int64_t CH = (int64_t) (((s16 ? 128ull : 256ull) << 20) / ((size_t) MR * es));
    CH = std::max<int64_t>(256, std::min<int64_t>(CH, NKV)) / 16 * 16;
    if (CH <= 0) CH = 16;
    for (int64_t j0 = 0; j0 < NKV; j0 += CH) {
        const int64_t nc = std::min<int64_t>(CH, NKV - j0);
        if (s16) {
            half * s = (half *) Sbuf;
            mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, MR, nc, E, half(1.0f), qh, E,
                                       kh + j0 * E, E, half(0.0f), s, nc);
            q.parallel_for(sycl::range<2>((size_t) B, (size_t) nc), [=](sycl::id<2> id) {
                const int64_t b = (int64_t) id[0], jj = (int64_t) id[1];
                float acc = 0.0f;
                for (int64_t h = 0; h < H; ++h) acc += sycl::fmax((float) s[(b * H + h) * nc + jj], 0.0f) * w[b * H + h];
                od[(j0 + jj) + b * NKV] = acc + (float) m[b * NKV + j0 + jj];
            });
        } else {
            float * s = (float *) Sbuf;
            mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans, MR, nc, E, 1.0f, qh, E,
                                       kh + j0 * E, E, 0.0f, s, nc);
            q.parallel_for(sycl::range<2>((size_t) B, (size_t) nc), [=](sycl::id<2> id) {
                const int64_t b = (int64_t) id[0], jj = (int64_t) id[1];
                float acc = 0.0f;
                for (int64_t h = 0; h < H; ++h) acc += sycl::fmax(s[(b * H + h) * nc + jj], 0.0f) * w[b * H + h];
                od[(j0 + jj) + b * NKV] = acc + (float) m[b * NKV + j0 + jj];
            });
        }
    }
}

// ---- the fused kernels' inputs: Q -> fp16 [B][H][E]; K -> fp16, padded with zero keys to NKVp (a multiple of 64):
// col_major = the keys' own layout; packed = per 16-key tile nt and 16-dim step ks a 16 x 16 VNNI tile, element (k2, n, p)
// at ((nt * 8 + ks) * 8 + k2) * 32 + n * 2 + p = key[16 nt + n][16 ks + 2 k2 + p]
static void li_prep(sycl::queue & q, const float * qf, const bf16 * k, int64_t B, int64_t NKV, int64_t NKVp, half * qh, half * kb) {
    q.parallel_for(sycl::range<1>((size_t) (B * H * E)), [=](sycl::id<1> i) { qh[i] = (half) qf[i]; });
    q.parallel_for(sycl::range<1>((size_t) (NKVp * E)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t j = i / E, e = i % E;
        const half v = j < NKV ? (half) (float) k[i] : half(0.0f);
#if LI_PACKED
        const int64_t nt = j / 16, n = j % 16, ks = e / 16, k2 = (e % 16) / 2, p = e % 2;
        kb[((nt * 8 + ks) * 8 + k2) * 32 + n * 2 + p] = v;
#else
        kb[i] = v;
#endif
    });
}

static inline void load_b(sycl::sub_group sg, jB & b, const half * kb, int64_t j0, int ks) {
#if LI_PACKED
    mx::joint_matrix_load(sg, b, gp(kb + ((j0 / 16) * 8 + ks) * 256), 32);
#else
    mx::joint_matrix_load(sg, b, gp(kb + j0 * E + ks * 16), E);
#endif
}

// the head sum of one 16-key tile's 4 accumulators (heads 0-7, 8-15, 16-23, 24-31): lane = key, element i = head 8 hg + i
#define LI_HSUM(s, x0, x1, x2, x3)                                                                         \
    {                                                                                                     \
        auto v0 = sycl::ext::oneapi::detail::get_wi_data(sg, x0);                                         \
        auto v1 = sycl::ext::oneapi::detail::get_wi_data(sg, x1);                                         \
        auto v2 = sycl::ext::oneapi::detail::get_wi_data(sg, x2);                                         \
        auto v3 = sycl::ext::oneapi::detail::get_wi_data(sg, x3);                                         \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                   \
            s += sycl::select_from_group(sg, wl0, i) * sycl::fmax((float) v0[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl0, 8 + i) * sycl::fmax((float) v1[i], 0.0f);               \
            s += sycl::select_from_group(sg, wl1, i) * sycl::fmax((float) v2[i], 0.0f);                   \
            s += sycl::select_from_group(sg, wl1, 8 + i) * sycl::fmax((float) v3[i], 0.0f);               \
        }                                                                                                 \
    }

// f2: NT = 2 (32 keys a step), A re-loaded every step; 8 accumulators (GRF128)
static void li_f2(sycl::queue & q, const half * qh, const half * kb, int64_t NKV, const float * w, const half * m, float * od,
                  int64_t B, int QW, int64_t KR) {
    const int64_t nkr = (NKV + KR - 1) / KR, nqb = (B + QW - 1) / QW;
    q.parallel_for(sycl::nd_range<1>((size_t) (nqb * nkr * QW * 16), (size_t) (QW * 16)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t kr = g % nkr, b = (g / nkr) * QW + (int64_t) sg.get_group_linear_id();
        if (b >= B) return;
        const float wl0 = w[b * H + lane], wl1 = w[b * H + 16 + lane];
        const half * qr = qh + b * H * E;
        const half * mr = m + b * NKV;
        float * orow = od + b * NKV;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja < NKV ? (float) mr[ja] : -INFINITY;
            const float mb = jb < NKV ? (float) mr[jb] : -INFINITY;
            if (sycl::all_of_group(sg, ma == -INFINITY && mb == -INFINITY)) {
                if (ja < NKV) orow[ja] = -INFINITY;
                if (jb < NKV) orow[jb] = -INFINITY;
                continue;
            }
            jC c00, c01, c10, c11, c20, c21, c30, c31;
            mx::joint_matrix_fill(sg, c00, 0.0f); mx::joint_matrix_fill(sg, c01, 0.0f);
            mx::joint_matrix_fill(sg, c10, 0.0f); mx::joint_matrix_fill(sg, c11, 0.0f);
            mx::joint_matrix_fill(sg, c20, 0.0f); mx::joint_matrix_fill(sg, c21, 0.0f);
            mx::joint_matrix_fill(sg, c30, 0.0f); mx::joint_matrix_fill(sg, c31, 0.0f);
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                jA a0, a1, a2, a3;
                jB b0, b1;
                mx::joint_matrix_load(sg, a0, gp(qr + 0 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a1, gp(qr + 1 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a2, gp(qr + 2 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a3, gp(qr + 3 * 8 * E + ks * 16), E);
                load_b(sg, b0, kb, j0, ks);
                load_b(sg, b1, kb, j0 + 16, ks);
                mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                mx::joint_matrix_mad(sg, c20, a2, b0, c20); mx::joint_matrix_mad(sg, c21, a2, b1, c21);
                mx::joint_matrix_mad(sg, c30, a3, b0, c30); mx::joint_matrix_mad(sg, c31, a3, b1, c31);
            }
            float s0 = 0.0f, s1 = 0.0f;
            LI_HSUM(s0, c00, c10, c20, c30)
            LI_HSUM(s1, c01, c11, c21, c31)
            if (ja < NKV) orow[ja] = s0 + ma;
            if (jb < NKV) orow[jb] = s1 + mb;
        }
    });
}

// f4: NT = 4 (64 keys a step), A re-loaded every step; 16 accumulators (GRF256)
static void li_f4(sycl::queue & q, const half * qh, const half * kb, int64_t NKV, const float * w, const half * m, float * od,
                  int64_t B, int QW, int64_t KR) {
    const int64_t nkr = (NKV + KR - 1) / KR, nqb = (B + QW - 1) / QW;
    q.parallel_for(sycl::nd_range<1>((size_t) (nqb * nkr * QW * 16), (size_t) (QW * 16)),
                   syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t kr = g % nkr, b = (g / nkr) * QW + (int64_t) sg.get_group_linear_id();
        if (b >= B) return;
        const float wl0 = w[b * H + lane], wl1 = w[b * H + 16 + lane];
        const half * qr = qh + b * H * E;
        const half * mr = m + b * NKV;
        float * orow = od + b * NKV;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 64) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane, jc = j0 + 32 + lane, jd = j0 + 48 + lane;
            const float ma = ja < NKV ? (float) mr[ja] : -INFINITY;
            const float mb = jb < NKV ? (float) mr[jb] : -INFINITY;
            const float mc = jc < NKV ? (float) mr[jc] : -INFINITY;
            const float md = jd < NKV ? (float) mr[jd] : -INFINITY;
            if (sycl::all_of_group(sg, ma == -INFINITY && mb == -INFINITY && mc == -INFINITY && md == -INFINITY)) {
                if (ja < NKV) orow[ja] = -INFINITY;
                if (jb < NKV) orow[jb] = -INFINITY;
                if (jc < NKV) orow[jc] = -INFINITY;
                if (jd < NKV) orow[jd] = -INFINITY;
                continue;
            }
            jC c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
            mx::joint_matrix_fill(sg, c00, 0.0f); mx::joint_matrix_fill(sg, c01, 0.0f);
            mx::joint_matrix_fill(sg, c02, 0.0f); mx::joint_matrix_fill(sg, c03, 0.0f);
            mx::joint_matrix_fill(sg, c10, 0.0f); mx::joint_matrix_fill(sg, c11, 0.0f);
            mx::joint_matrix_fill(sg, c12, 0.0f); mx::joint_matrix_fill(sg, c13, 0.0f);
            mx::joint_matrix_fill(sg, c20, 0.0f); mx::joint_matrix_fill(sg, c21, 0.0f);
            mx::joint_matrix_fill(sg, c22, 0.0f); mx::joint_matrix_fill(sg, c23, 0.0f);
            mx::joint_matrix_fill(sg, c30, 0.0f); mx::joint_matrix_fill(sg, c31, 0.0f);
            mx::joint_matrix_fill(sg, c32, 0.0f); mx::joint_matrix_fill(sg, c33, 0.0f);
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                jA a0, a1, a2, a3;
                jB b0, b1, b2, b3;
                mx::joint_matrix_load(sg, a0, gp(qr + 0 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a1, gp(qr + 1 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a2, gp(qr + 2 * 8 * E + ks * 16), E);
                mx::joint_matrix_load(sg, a3, gp(qr + 3 * 8 * E + ks * 16), E);
                load_b(sg, b0, kb, j0, ks);
                load_b(sg, b1, kb, j0 + 16, ks);
                load_b(sg, b2, kb, j0 + 32, ks);
                load_b(sg, b3, kb, j0 + 48, ks);
                mx::joint_matrix_mad(sg, c00, a0, b0, c00); mx::joint_matrix_mad(sg, c01, a0, b1, c01);
                mx::joint_matrix_mad(sg, c02, a0, b2, c02); mx::joint_matrix_mad(sg, c03, a0, b3, c03);
                mx::joint_matrix_mad(sg, c10, a1, b0, c10); mx::joint_matrix_mad(sg, c11, a1, b1, c11);
                mx::joint_matrix_mad(sg, c12, a1, b2, c12); mx::joint_matrix_mad(sg, c13, a1, b3, c13);
                mx::joint_matrix_mad(sg, c20, a2, b0, c20); mx::joint_matrix_mad(sg, c21, a2, b1, c21);
                mx::joint_matrix_mad(sg, c22, a2, b2, c22); mx::joint_matrix_mad(sg, c23, a2, b3, c23);
                mx::joint_matrix_mad(sg, c30, a3, b0, c30); mx::joint_matrix_mad(sg, c31, a3, b1, c31);
                mx::joint_matrix_mad(sg, c32, a3, b2, c32); mx::joint_matrix_mad(sg, c33, a3, b3, c33);
            }
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            LI_HSUM(s0, c00, c10, c20, c30)
            LI_HSUM(s1, c01, c11, c21, c31)
            LI_HSUM(s2, c02, c12, c22, c32)
            LI_HSUM(s3, c03, c13, c23, c33)
            if (ja < NKV) orow[ja] = s0 + ma;
            if (jb < NKV) orow[jb] = s1 + mb;
            if (jc < NKV) orow[jc] = s2 + mc;
            if (jd < NKV) orow[jd] = s3 + md;
        }
    });
}

// f2q: NT = 2, the query's 32 A tiles loaded once into registers (named: joint_matrix arrays miscompile on IGC); GRF256
#define LI_AQ(hg, ks) jA a##hg##ks; mx::joint_matrix_load(sg, a##hg##ks, gp(qr + (hg) * 8 * E + (ks) * 16), E);
#define LI_AQK(ks) LI_AQ(0, ks) LI_AQ(1, ks) LI_AQ(2, ks) LI_AQ(3, ks)
#define LI_STEPQ(ks)                                                                                       \
    {                                                                                                     \
        jB b0, b1;                                                                                        \
        load_b(sg, b0, kb, j0, ks);                                                                       \
        load_b(sg, b1, kb, j0 + 16, ks);                                                                  \
        mx::joint_matrix_mad(sg, c00, a0##ks, b0, c00); mx::joint_matrix_mad(sg, c01, a0##ks, b1, c01);   \
        mx::joint_matrix_mad(sg, c10, a1##ks, b0, c10); mx::joint_matrix_mad(sg, c11, a1##ks, b1, c11);   \
        mx::joint_matrix_mad(sg, c20, a2##ks, b0, c20); mx::joint_matrix_mad(sg, c21, a2##ks, b1, c21);   \
        mx::joint_matrix_mad(sg, c30, a3##ks, b0, c30); mx::joint_matrix_mad(sg, c31, a3##ks, b1, c31);   \
    }
static void li_f2q(sycl::queue & q, const half * qh, const half * kb, int64_t NKV, const float * w, const half * m, float * od,
                   int64_t B, int QW, int64_t KR) {
    const int64_t nkr = (NKV + KR - 1) / KR, nqb = (B + QW - 1) / QW;
    q.parallel_for(sycl::nd_range<1>((size_t) (nqb * nkr * QW * 16), (size_t) (QW * 16)),
                   syclex::properties{sycl::ext::intel::experimental::grf_size<256>},
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t g = (int64_t) it.get_group(0);
        const int64_t kr = g % nkr, b = (g / nkr) * QW + (int64_t) sg.get_group_linear_id();
        if (b >= B) return;
        const float wl0 = w[b * H + lane], wl1 = w[b * H + 16 + lane];
        const half * qr = qh + b * H * E;
        const half * mr = m + b * NKV;
        float * orow = od + b * NKV;
        LI_AQK(0) LI_AQK(1) LI_AQK(2) LI_AQK(3) LI_AQK(4) LI_AQK(5) LI_AQK(6) LI_AQK(7)
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja < NKV ? (float) mr[ja] : -INFINITY;
            const float mb = jb < NKV ? (float) mr[jb] : -INFINITY;
            if (sycl::all_of_group(sg, ma == -INFINITY && mb == -INFINITY)) {
                if (ja < NKV) orow[ja] = -INFINITY;
                if (jb < NKV) orow[jb] = -INFINITY;
                continue;
            }
            jC c00, c01, c10, c11, c20, c21, c30, c31;
            mx::joint_matrix_fill(sg, c00, 0.0f); mx::joint_matrix_fill(sg, c01, 0.0f);
            mx::joint_matrix_fill(sg, c10, 0.0f); mx::joint_matrix_fill(sg, c11, 0.0f);
            mx::joint_matrix_fill(sg, c20, 0.0f); mx::joint_matrix_fill(sg, c21, 0.0f);
            mx::joint_matrix_fill(sg, c30, 0.0f); mx::joint_matrix_fill(sg, c31, 0.0f);
            LI_STEPQ(0) LI_STEPQ(1) LI_STEPQ(2) LI_STEPQ(3) LI_STEPQ(4) LI_STEPQ(5) LI_STEPQ(6) LI_STEPQ(7)
            float s0 = 0.0f, s1 = 0.0f;
            LI_HSUM(s0, c00, c10, c20, c30)
            LI_HSUM(s1, c01, c11, c21, c31)
            if (ja < NKV) orow[ja] = s0 + ma;
            if (jb < NKV) orow[jb] = s1 + mb;
        }
    });
}

template <typename F> static double time_ms(sycl::queue & q, F && f, int reps) {
    f();
    f();
    q.wait();
    const double t0 = now_ms();
    for (int i = 0; i < reps; ++i) f();
    q.wait();
    return (now_ms() - t0) / reps;
}

struct Cmp { double maxd, maxr; int64_t inf_bad, n; };
static Cmp cmp(const std::vector<float> & a, const std::vector<float> & r) {
    Cmp c{0, 0, 0, 0};
    for (size_t i = 0; i < r.size(); ++i) {
        const bool fa = std::isfinite(a[i]), fr = std::isfinite(r[i]);
        if (fa != fr || (!fa && a[i] != r[i])) { ++c.inf_bad; continue; }
        if (!fr) continue;
        c.maxd = std::max(c.maxd, (double) std::fabs(a[i] - r[i]));
        c.maxr = std::max(c.maxr, (double) std::fabs(r[i]));
        ++c.n;
    }
    return c;
}

int main(int argc, char ** argv) {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("%s, LI_PACKED=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), LI_PACKED);
    const int64_t B = std::getenv("LI_B") ? std::atoll(std::getenv("LI_B")) : 512;
    const int QW = std::getenv("LI_QW") ? std::atoi(std::getenv("LI_QW")) : 8;
    const int64_t KR = std::getenv("LI_KR") ? std::atoll(std::getenv("LI_KR")) : 1024;
    const int reps = std::getenv("LI_REPS") ? std::atoi(std::getenv("LI_REPS")) : 10;
    std::vector<std::pair<int64_t, int64_t>> shapes;   // (cap, visible rows of the span's last query)
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const size_t c = s.find(':');
        shapes.push_back({std::atoll(s.substr(0, c).c_str()), c == std::string::npos ? std::atoll(s.c_str()) : std::atoll(s.substr(c + 1).c_str())});
    }
    if (shapes.empty()) shapes = {{1024, 1000}, {4096, 3900}, {8192, 4400}, {8192, 8000}, {16384, 8400}, {16384, 15800}, {65536, 60000}, {262144, 250000}};
    std::mt19937 rng(1017);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::printf("B %lld queries x %lld heads x %lld dims; fused: QW %d queries a work-group, KR %lld keys a work-group\n",
                (long long) B, (long long) H, (long long) E, QW, (long long) KR);
    std::printf("%8s %8s | %9s %9s | %9s %9s %9s | max|d|/max|r|: s16, f2, f4, f2q (ref vs cpu) | -inf bad\n", "cap", "visible",
                "ref ms", "s16 ms", "f2 ms", "f4 ms", "f2q ms");
    for (auto [NKV, V] : shapes) {
        const int64_t NKVp = (NKV + 63) / 64 * 64;
        std::vector<float> hq((size_t) (B * H * E)), hw((size_t) (B * H));
        std::vector<bf16> hk((size_t) (NKV * E));
        std::vector<half> hm((size_t) (B * NKV));
        for (auto & x : hq) x = nd(rng);
        for (auto & x : hk) x = (bf16) nd(rng);
        for (auto & x : hw) x = nd(rng) / 64.0f;
        for (int64_t b = 0; b < B; ++b) {   // a ratio-2 span: the span's B queries see V - (B - 1 - b) / 2 rows
            const int64_t nv = std::max<int64_t>(1, V - (B - 1 - b) / 2);
            for (int64_t j = 0; j < NKV; ++j) hm[(size_t) (b * NKV + j)] = j < nv ? half(0.0f) : half(-INFINITY);
        }
        float * dq = sycl::malloc_device<float>(hq.size(), q);
        bf16 * dk = sycl::malloc_device<bf16>(hk.size(), q);
        float * dw = sycl::malloc_device<float>(hw.size(), q);
        half * dm = sycl::malloc_device<half>(hm.size(), q);
        float * dout = sycl::malloc_device<float>((size_t) (B * NKV), q);
        half * qh = sycl::malloc_device<half>((size_t) (B * H * E), q);
        half * kh = sycl::malloc_device<half>((size_t) (NKVp * E), q);
        void * S = sycl::malloc_device((size_t) 256 << 20, q);
        q.memcpy(dq, hq.data(), hq.size() * 4);
        q.memcpy(dk, hk.data(), hk.size() * 2);
        q.memcpy(dw, hw.data(), hw.size() * 4);
        q.memcpy(dm, hm.data(), hm.size() * 2).wait();
        std::vector<float> ref((size_t) (B * NKV)), got((size_t) (B * NKV));
        auto run_ref = [&] { li_ref(q, dq, dk, dw, dm, dout, B, NKV, qh, kh, S, false); };
        auto run_s16 = [&] { li_ref(q, dq, dk, dw, dm, dout, B, NKV, qh, kh, S, true); };
        auto run_f2 = [&] { li_prep(q, dq, dk, B, NKV, NKVp, qh, kh); li_f2(q, qh, kh, NKV, dw, dm, dout, B, QW, KR); };
        auto run_f4 = [&] { li_prep(q, dq, dk, B, NKV, NKVp, qh, kh); li_f4(q, qh, kh, NKV, dw, dm, dout, B, QW, KR); };
        auto run_f2q = [&] { li_prep(q, dq, dk, B, NKV, NKVp, qh, kh); li_f2q(q, qh, kh, NKV, dw, dm, dout, B, QW, KR); };
        run_ref();
        q.memcpy(ref.data(), dout, ref.size() * 4).wait();
        // the reference against the CPU on sampled entries (fp16-rounded inputs, double sums)
        double cpu_d = 0, cpu_r = 0;
        for (int t = 0; t < 64; ++t) {
            const int64_t b = (t * 97) % B, j = std::min<int64_t>(NKV - 1, (int64_t) ((t * 7919ll) % std::max<int64_t>(1, V)));
            double s = 0;
            for (int64_t h = 0; h < H; ++h) {
                double d = 0;
                for (int64_t e = 0; e < E; ++e) d += (double) (float) (half) hq[(size_t) ((b * H + h) * E + e)] * (double) (float) (half) (float) hk[(size_t) (j * E + e)];
                s += std::max(d, 0.0) * (double) hw[(size_t) (b * H + h)];
            }
            s += (double) (float) hm[(size_t) (b * NKV + j)];
            const float r = ref[(size_t) (b * NKV + j)];
            if (std::isfinite(r) && std::isfinite(s)) { cpu_d = std::max(cpu_d, std::fabs(r - s)); cpu_r = std::max(cpu_r, std::fabs(s)); }
        }
        auto check = [&](auto && f) { q.memset(dout, 0x7f, (size_t) (B * NKV) * 4).wait(); f(); q.memcpy(got.data(), dout, got.size() * 4).wait(); return cmp(got, ref); };
        const Cmp c16 = check(run_s16), c2 = check(run_f2), c4 = check(run_f4), c2q = check(run_f2q);
        const double t_ref = time_ms(q, run_ref, reps), t_s16 = time_ms(q, run_s16, reps);
        const double t_f2 = time_ms(q, run_f2, reps), t_f4 = time_ms(q, run_f4, reps), t_f2q = time_ms(q, run_f2q, reps);
        auto rel = [](const Cmp & c) { return c.maxr > 0 ? c.maxd / c.maxr : 0.0; };
        std::printf("%8lld %8lld | %9.3f %9.3f | %9.3f %9.3f %9.3f | %.1e %.1e %.1e %.1e (%.1e) | %lld %lld %lld %lld\n", (long long) NKV,
                    (long long) V, t_ref, t_s16, t_f2, t_f4, t_f2q, rel(c16), rel(c2), rel(c4), rel(c2q), cpu_r > 0 ? cpu_d / cpu_r : 0.0,
                    (long long) c16.inf_bad, (long long) c2.inf_bad, (long long) c4.inf_bad, (long long) c2q.inf_bad);
        std::fflush(stdout);
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dw, q); sycl::free(dm, q); sycl::free(dout, q);
        sycl::free(qh, q); sycl::free(kh, q); sycl::free(S, q);
    }
    return 0;
}
