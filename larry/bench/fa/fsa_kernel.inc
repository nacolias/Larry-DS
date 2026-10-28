// fsa_kernel.inc - larry (#3, 2026-10-28): DeepSeek-V4.1's prompt attention for a span's queries as ONE kernel - each
// query's window rows and selected compressed rows read where they lie (no gathered [n, 656, 576] copy), its 64 MQA
// heads' scores, an online softmax with the head's sink, and the weighted sum of the same rows, fp16 x fp16 -> f32 on XMX.
//
// wcopy_run first writes the span's window rows in fp16 by position (row i = the first query's position - swa + 1 + i),
// so query t's window is rows t .. t + swa - 1 of it, read in place.  Then a work-group = one query at a time (a
// persistent loop over the span's queries: nslot groups, query t on group t % nslot), 16 sub-groups, GRF 256.  Per query:
// its Q scaled to fp16 into the group's slot of a device scratch (64 KB, block reads / writes), then its M keys in chunks
// of 128 - chunk 0 the window (in place), later chunks its selected compressed rows converted to fp16 into the slot (128
// KB, block reads / writes; the next chunk's rows prefetched to L1 meanwhile; a slot is reused query after query, so it
// stays in L2) - each chunk:
//   QK^T  sub-group (hg, kq): heads 16 hg .. +15 x keys 32 kq .. +31 of the chunk, 2 x 2 accumulators over the 512 dims
//         - A = Q rows, B = K^T straight from the key rows (col_major 2D block loads), the next step's tiles loaded first
//   softmax  the chunk's max a head from the 4 key quarters (local memory), P = exp(s - m) in fp16 into local memory,
//         the rescale factor and the partial sums
//   PV    sub-group (hh, dg): heads 32 hh .. +31 x dims 64 dg .. +63 (4 x 4 accumulators, the query's O in registers):
//         A = P tiles from local memory, B = V tiles from the same key rows (row_major: VNNI-transforming block loads)
// and at the end O / (l + exp(sink - m)) in fp16.  Rounding as the oneDNN path: Q * scale and the keys in fp16, scores
// and sums in f32, P in fp16 (here before the division by the sum, there after), O accumulated in f32.
// bench/fa/test_fsa.cpp: within the oneDNN path's error of a double reference; 512 queries at 30K 1.44 ms against the
// service path's 2.37 (KGATHER + Q' / mask + oneDNN SDPA + the output copy).
//
// Inputs (the dense graph's tensors):
//   q      F32 [n][64][512] (rotated)          win   F32 window rows, wnb bytes apart; widx I32 [n][swa] rows of win
//   comp   BF16 compressed rows, cnb bytes apart (nullptr: window only); sel I32 [n][ldsel] (the first M - swa used)
//   mask   F32 [n][ldm]: M values a query (window rows, then compressed), 0 or -inf  (a masked row is never read)
//   sinks  F32 [64]                             out   F16 [n][64][512]
#include <cmath>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/__spirv/spirv_ops.hpp>

namespace fsa {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using half = sycl::half;

constexpr int NH = 64, DH = 512, CK = 128;   // heads, head dims, keys a chunk
constexpr int NSG = 16, WGS = NSG * 16;
constexpr size_t QH = (size_t) NH * DH, KSZ = (size_t) CK * DH, SLOT = QH + KSZ;   // halves a slot

struct Args {
    const float * q = nullptr;
    const char * win = nullptr;
    size_t wnb = 0;
    const int32_t * widx = nullptr;
    int swa = 128;
    const char * comp = nullptr;
    size_t cnb = 0;
    const int32_t * sel = nullptr;
    int ldsel = 0;
    const float * mask = nullptr;
    int ldm = 0;
    int M = 0;   // keys a query: swa (+ the compressed ones); a multiple of CK
    const float * sinks = nullptr;
    float scale = 1.0f;
    half * out = nullptr;
    int n = 0;
    half * scratch = nullptr;   // nslot * SLOT halves
    half * wcopy = nullptr;     // (n + swa - 1) x DH halves: the span's window rows in fp16 by position (wcopy_run)
    int nslot = 0;
    int skip = 0;   // timing only: 1 the rows' loads, 2 QK^T, 4 the softmax, 8 PV, 16 the output
};

typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
// a sub-group block read / write of 8 dwords a lane from / to global memory (striped: lane l's dword k = the block's
// dword 16 k + l); p 16-byte aligned
inline u32x8 blk_rd(const uint32_t * p) {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupBlockReadINTEL<u32x8>(
        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(p).get_decorated());
#else
    (void) p;
    return u32x8(0);
#endif
}
inline void blk_wr(uint32_t * p, u32x8 v) {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_SubgroupBlockWriteINTEL(
        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(p).get_decorated(), v);
#else
    (void) p;
    (void) v;
#endif
}
typedef uint64_t u64x8 __attribute__((ext_vector_type(8)));
inline u64x8 blk_rd64(const uint64_t * p) {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupBlockReadINTEL<u64x8>(
        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(p).get_decorated());
#else
    (void) p;
    return u64x8(0);
#endif
}
// two F32 values (a qword) times s as two fp16 values (a dword)
inline uint32_t f2h2(uint64_t w, float sc) {
    const sycl::vec<sycl::half, 2> hp((sycl::half) (sycl::bit_cast<float>((uint32_t) w) * sc), (sycl::half) (sycl::bit_cast<float>((uint32_t) (w >> 32)) * sc));
    return sycl::bit_cast<uint32_t>(hp);
}
// two BF16 values (a dword) as two fp16 values
inline uint32_t bf2h2(uint32_t w) {
    const sycl::vec<sycl::half, 2> hp((sycl::half) sycl::bit_cast<float>(w << 16), (sycl::half) sycl::bit_cast<float>(w & 0xffff0000u));
    return sycl::bit_cast<uint32_t>(hp);
}

using A_t = mx::joint_matrix<sycl::sub_group, half, mx::use::a, 8, 16, mx::layout::row_major>;
using Bc_t = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::col_major>;
using Br_t = mx::joint_matrix<sycl::sub_group, half, mx::use::b, 16, 16, mx::layout::row_major>;
using C_t = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;

inline size_t slot_bytes() { return SLOT * sizeof(half); }

// the 16 O accumulators of a PV sub-group as named objects (arrays of joint_matrix miscompile on this IGC)
#define FSA_O16(X) X(0, 0) X(0, 1) X(0, 2) X(0, 3) X(1, 0) X(1, 1) X(1, 2) X(1, 3) X(2, 0) X(2, 1) X(2, 2) X(2, 3) X(3, 0) X(3, 1) X(3, 2) X(3, 3)

// the span's window rows in fp16, row i = position (first query's) - swa + 1 + i: rows i < swa - 1 from query 0's slots,
// then each query's own row (its last slot); a masked slot (before position 0) as zeros.  Query t's window = rows t .. t +
// swa - 1, in slot order.  One sub-group a row (block reads / writes).
inline sycl::event wcopy_run(sycl::queue & q, const Args & a) {
    const Args A = a;
    const int rows = A.n + A.swa - 1;
    return q.parallel_for(sycl::nd_range<1>((size_t) (rows + 3) / 4 * 64, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        constexpr auto bprops = syclex::properties{syclex::data_placement_striped, syclex::full_group};
        const int i = (int) (it.get_global_id(0) / 16);
        if (i >= rows) return;
        const int t = i < A.swa - 1 ? 0 : i - (A.swa - 1), j = i < A.swa - 1 ? i : A.swa - 1;
        const bool ok = A.mask[(size_t) t * A.ldm + j] == 0.0f;
        sycl::vec<uint32_t, 16> o(0u);
        if (ok) {
            sycl::vec<uint64_t, 16> f;
            syclex::group_load(sg, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                       reinterpret_cast<const uint64_t *>(A.win + (size_t) A.widx[(size_t) t * A.swa + j] * A.wnb)), f, bprops);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint64_t w = f[k];
                const sycl::vec<half, 2> hp((half) sycl::bit_cast<float>((uint32_t) w), (half) sycl::bit_cast<float>((uint32_t) (w >> 32)));
                o[k] = sycl::bit_cast<uint32_t>(hp);
            }
        }
        syclex::group_store(sg, o, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                       reinterpret_cast<uint32_t *>(A.wcopy + (size_t) i * DH)), bprops);
    });
}

template <int SKIP>
inline sycl::event run_t(sycl::queue & q, const Args & a) {
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<half, 1> Ps(sycl::range<1>((size_t) NH * CK), h);   // P [64 heads][128 keys]
        // lmax[4][64], psum[4][64], mrun[64], lrun[64], mnew[64], alpha[64], maskc[128]
        sycl::local_accessor<float, 1> red(sycl::range<1>(4 * NH + 4 * NH + 4 * NH + CK), h);
        const Args A = a;
        h.parallel_for(sycl::nd_range<1>((size_t) A.nslot * WGS, WGS),
                       sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            constexpr auto bprops = syclex::properties{syclex::data_placement_striped, syclex::full_group};
            const int wg = (int) it.get_group(0), lid = (int) it.get_local_id(0);
            const int s = (int) sg.get_group_linear_id(), ln = (int) sg.get_local_linear_id();
            half * Qh = A.scratch + (size_t) wg * SLOT;
            half * Ks = Qh + QH;
            auto gQ = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Qh);
            auto lP = Ps.template get_multi_ptr<sycl::access::decorated::no>();
            float * lmax = &red[0];
            float * psum = lmax + 4 * NH;
            float * mrun = psum + 4 * NH;
            float * lrun = mrun + NH;
            float * mnew = lrun + NH;
            float * alpha = mnew + NH;
            float * maskc = alpha + NH;
            const int hg = s & 3, kq = s >> 2;   // QK^T / softmax: heads 16 hg .., keys 32 kq .. of a chunk
            const int hh = s & 1, dg = s >> 1;   // PV: heads 32 hh .., dims 64 dg ..
            const int nch = A.M / CK;
            for (int t = wg; t < A.n; t += A.nslot) {
                // ---- this query's Q * scale in fp16 into the slot (sub-group s: heads 4 s .. 4 s + 3, a row as two
                // 8-qword block reads and two 8-dword block writes); m = -inf, l = 0
                {
                    const float * qt = A.q + (size_t) t * QH;
#pragma unroll
                    for (int u = 0; u < NH / NSG; ++u) {
                        const int hd = (NH / NSG) * s + u;
                        const uint64_t * src = reinterpret_cast<const uint64_t *>(qt + (size_t) hd * DH);
                        uint32_t * dst = reinterpret_cast<uint32_t *>(Qh + (size_t) hd * DH);
#pragma unroll
                        for (int hf = 0; hf < 2; ++hf) {   // qwords 128 hf + 16 k + lane -> dwords 128 hf + 16 k + lane
                            const u64x8 a = blk_rd64(src + 128 * hf);
                            u32x8 o;
#pragma unroll
                            for (int k = 0; k < 8; ++k) o[k] = f2h2(a[k], A.scale);
                            blk_wr(dst + 128 * hf, o);
                        }
                    }
                    if (lid < NH) { mrun[lid] = -INFINITY; lrun[lid] = 0.0f; }
                }
                sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
#define FSA_DECL(i, j) C_t o##i##j; mx::joint_matrix_fill(sg, o##i##j, 0.0f);
                FSA_O16(FSA_DECL)
#undef FSA_DECL
                for (int c = 0; c < nch; ++c) {
                    // the chunk's key rows: chunk 0 = the window (rows t .. t + 127 of the window copy, in place), later
                    // chunks the selected compressed rows staged into the slot
                    const half * kbase = c == 0 && A.swa == CK ? A.wcopy + (size_t) t * DH : Ks;
                    auto gK = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kbase);
                    if (kbase == Ks) {
                    // ---- the chunk's 128 compressed rows in fp16 into the slot: sub-group s takes rows s, s + 16, ..,
                    // two at a time, each read and written as two sub-group blocks of 8 dwords a lane (a dword = two
                    // values: BF16 pairs in, fp16 pairs out, the same positions).  A masked row is written as zeros.
                    for (int r0 = s; r0 < CK; r0 += 2 * NSG) {
                        const uint32_t * src[2];
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            const int m = c * CK + r0 + NSG * u;
                            src[u] = nullptr;
                            if (!(SKIP & 1) && A.mask[(size_t) t * A.ldm + m] == 0.0f)
                                src[u] = reinterpret_cast<const uint32_t *>(A.comp + (size_t) A.sel[(size_t) t * A.ldsel + (m - A.swa)] * A.cnb);
                        }
                        u32x8 lo[2], hi[2];
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            lo[u] = 0; hi[u] = 0;
                            if (src[u]) { lo[u] = blk_rd(src[u]); hi[u] = blk_rd(src[u] + 128); }
                        }
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            u32x8 a = lo[u], b = hi[u];
#pragma unroll
                            for (int k = 0; k < 8; ++k) { a[k] = bf2h2(a[k]); b[k] = bf2h2(b[k]); }
                            uint32_t * dst = reinterpret_cast<uint32_t *>(Ks + (size_t) (r0 + NSG * u) * DH);
                            blk_wr(dst, a);
                            blk_wr(dst + 128, b);
                        }
                    }
                    sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
                    }
                    if constexpr (!(SKIP & 1)) {   // the next staged chunk's rows toward L1 while this one computes
                        const int tn = c + 1 < nch ? t : t + A.nslot, cn = c + 1 < nch ? c + 1 : 1;
                        if (tn < A.n && cn < nch) {
#pragma unroll
                            for (int u = 0; u < CK / NSG; ++u) {
                                const int m = cn * CK + s + NSG * u;
                                if (A.mask[(size_t) tn * A.ldm + m] == 0.0f)
                                    syclex::prefetch(A.comp + (size_t) A.sel[(size_t) tn * A.ldsel + (m - A.swa)] * A.cnb + ln * 64, 64,
                                                     syclex::properties{syclex::prefetch_hint_L1});
                            }
                        }
                    }
                    // ---- QK^T: heads 16 hg .. +15 x keys 32 kq .. +31, the 512 dims in 32 steps
                    C_t s00, s01, s10, s11;
                    mx::joint_matrix_fill(sg, s00, 0.0f); mx::joint_matrix_fill(sg, s01, 0.0f);
                    mx::joint_matrix_fill(sg, s10, 0.0f); mx::joint_matrix_fill(sg, s11, 0.0f);
                    {
                        A_t a0, a1, c0, c1;
                        Bc_t b0, b1, d0, d1;
                        auto ldq = [&](A_t & x0, A_t & x1, Bc_t & y0, Bc_t & y1, int ks) {
                            mx::joint_matrix_load(sg, x0, gQ + (size_t) (16 * hg) * DH + 16 * ks, DH);
                            mx::joint_matrix_load(sg, x1, gQ + (size_t) (16 * hg + 8) * DH + 16 * ks, DH);
                            mx::joint_matrix_load(sg, y0, gK + (size_t) (32 * kq) * DH + 16 * ks, DH);
                            mx::joint_matrix_load(sg, y1, gK + (size_t) (32 * kq + 16) * DH + 16 * ks, DH);
                        };
                        if (!(SKIP & 2)) ldq(a0, a1, b0, b1, 0);
                        for (int ks = 0; ks < ((SKIP & 2) ? 0 : DH / 16); ks += 2) {
                            ldq(c0, c1, d0, d1, ks + 1);
                            mx::joint_matrix_mad(sg, s00, a0, b0, s00);
                            mx::joint_matrix_mad(sg, s01, a0, b1, s01);
                            mx::joint_matrix_mad(sg, s10, a1, b0, s10);
                            mx::joint_matrix_mad(sg, s11, a1, b1, s11);
                            if (ks + 2 < DH / 16) ldq(a0, a1, b0, b1, ks + 2);
                            mx::joint_matrix_mad(sg, s00, c0, d0, s00);
                            mx::joint_matrix_mad(sg, s01, c0, d1, s01);
                            mx::joint_matrix_mad(sg, s10, c1, d0, s10);
                            mx::joint_matrix_mad(sg, s11, c1, d1, s11);
                        }
                    }
                    // the scores with the keys' mask (a lane = a key, element i = a head); each head's max over the 32 keys
                    auto w00 = sycl::ext::oneapi::detail::get_wi_data(sg, s00);
                    auto w01 = sycl::ext::oneapi::detail::get_wi_data(sg, s01);
                    auto w10 = sycl::ext::oneapi::detail::get_wi_data(sg, s10);
                    auto w11 = sycl::ext::oneapi::detail::get_wi_data(sg, s11);
                    const float mk0 = A.mask[(size_t) t * A.ldm + c * CK + 32 * kq + ln], mk1 = A.mask[(size_t) t * A.ldm + c * CK + 32 * kq + 16 + ln];
                    float v00[8], v01[8], v10[8], v11[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        v00[i] = (float) w00[i] + mk0; v01[i] = (float) w01[i] + mk1;
                        v10[i] = (float) w10[i] + mk0; v11[i] = (float) w11[i] + mk1;
                        const float x0 = sycl::reduce_over_group(sg, sycl::fmax(v00[i], v01[i]), sycl::maximum<float>());
                        const float x1 = sycl::reduce_over_group(sg, sycl::fmax(v10[i], v11[i]), sycl::maximum<float>());
                        if (ln == i) { lmax[kq * NH + 16 * hg + i] = x0; lmax[kq * NH + 16 * hg + 8 + i] = x1; }
                    }
                    sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
                    // ---- softmax: the chunk's max a head, P = exp(s - m) in fp16, the partial sums, the rescale factor
                    if (!(SKIP & 4))
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            const int hd = 16 * hg + 8 * u + i;
                            const float mo = mrun[hd];
                            const float mc = sycl::fmax(sycl::fmax(lmax[hd], lmax[NH + hd]), sycl::fmax(lmax[2 * NH + hd], lmax[3 * NH + hd]));
                            const float mn = sycl::fmax(mo, mc);
                            const float a0 = u ? v10[i] : v00[i], a1 = u ? v11[i] : v01[i];
                            const half p0 = (half) (mn == -INFINITY ? 0.0f : sycl::exp(a0 - mn));
                            const half p1 = (half) (mn == -INFINITY ? 0.0f : sycl::exp(a1 - mn));
                            Ps[(size_t) hd * CK + 32 * kq + ln] = p0;
                            Ps[(size_t) hd * CK + 32 * kq + 16 + ln] = p1;
                            const float ps = sycl::reduce_over_group(sg, (float) p0 + (float) p1, sycl::plus<float>());
                            if (ln == 0) {
                                psum[kq * NH + hd] = ps;
                                if (kq == 0) { mnew[hd] = mn; alpha[hd] = mn == -INFINITY ? 1.0f : sycl::exp(mo - mn); }
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
                    // ---- PV: O (heads 32 hh .. +31 x dims 64 dg .. +63) rescaled, then += P [32 x 128] . V [128 x 64]
                    if (dg == 0 && ln < 16) {
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            const int hd = 32 * hh + 16 * u + ln;
                            lrun[hd] = alpha[hd] * lrun[hd] + psum[hd] + psum[NH + hd] + psum[2 * NH + hd] + psum[3 * NH + hd];
                            mrun[hd] = mnew[hd];
                        }
                    }
                    {
                        float al[4][8];
#pragma unroll
                        for (int i = 0; i < 4; ++i)
#pragma unroll
                            for (int e = 0; e < 8; ++e) al[i][e] = alpha[32 * hh + 8 * i + e];
#define FSA_SCALE(i, j)                                                              \
                        {                                                            \
                            auto wo = sycl::ext::oneapi::detail::get_wi_data(sg, o##i##j); \
                            _Pragma("unroll") for (int e = 0; e < 8; ++e) wo[e] = (float) wo[e] * al[i][e]; \
                        }
                        FSA_O16(FSA_SCALE)
#undef FSA_SCALE
                    }
                    {
                        Br_t v0, v1, v2, v3, w0, w1, w2, w3;
                        auto ldv = [&](Br_t & y0, Br_t & y1, Br_t & y2, Br_t & y3, int kk) {
                            mx::joint_matrix_load(sg, y0, gK + (size_t) (16 * kk) * DH + 64 * dg + 0, DH);
                            mx::joint_matrix_load(sg, y1, gK + (size_t) (16 * kk) * DH + 64 * dg + 16, DH);
                            mx::joint_matrix_load(sg, y2, gK + (size_t) (16 * kk) * DH + 64 * dg + 32, DH);
                            mx::joint_matrix_load(sg, y3, gK + (size_t) (16 * kk) * DH + 64 * dg + 48, DH);
                        };
                        auto pvstep = [&](Br_t & y0, Br_t & y1, Br_t & y2, Br_t & y3, int kk) {
                            A_t p0, p1, p2, p3;
                            mx::joint_matrix_load(sg, p0, lP + (size_t) (32 * hh + 0) * CK + 16 * kk, CK);
                            mx::joint_matrix_load(sg, p1, lP + (size_t) (32 * hh + 8) * CK + 16 * kk, CK);
                            mx::joint_matrix_load(sg, p2, lP + (size_t) (32 * hh + 16) * CK + 16 * kk, CK);
                            mx::joint_matrix_load(sg, p3, lP + (size_t) (32 * hh + 24) * CK + 16 * kk, CK);
#define FSA_MAD(i, j) mx::joint_matrix_mad(sg, o##i##j, p##i, y##j, o##i##j);
                            FSA_O16(FSA_MAD)
#undef FSA_MAD
                        };
                        if (!(SKIP & 8)) ldv(v0, v1, v2, v3, 0);
                        for (int kk = 0; kk < ((SKIP & 8) ? 0 : CK / 16); kk += 2) {
                            ldv(w0, w1, w2, w3, kk + 1);
                            pvstep(v0, v1, v2, v3, kk);
                            if (kk + 2 < CK / 16) ldv(v0, v1, v2, v3, kk + 2);
                            pvstep(w0, w1, w2, w3, kk + 1);
                        }
                    }
                    sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
                }
                // ---- O / (l + exp(sink - m)) -> fp16
                {
                    half * ot = A.out + (size_t) t * QH;
                    float rd[4][8];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int hd = 32 * hh + 8 * i + e;
                            rd[i][e] = 1.0f / (lrun[hd] + sycl::exp(A.sinks[hd] - mrun[hd]));
                        }
#define FSA_OUT(i, j)                                                                                     \
                    {                                                                                     \
                        auto wo = sycl::ext::oneapi::detail::get_wi_data(sg, o##i##j);                   \
                        _Pragma("unroll") for (int e = 0; e < 8; ++e)                                     \
                            ot[(size_t) (32 * hh + 8 * i + e) * DH + 64 * dg + 16 * j + ln] = (half) ((float) wo[e] * rd[i][e]); \
                    }
                    if (!(SKIP & 16)) { FSA_O16(FSA_OUT) }
#undef FSA_OUT
                }
                sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
            }
        });
    });
}
#undef FSA_O16

// the whole op: the window copy, then the attention (FSA_TIMING builds, Args::skip > 0: a timing variant of the attention)
inline sycl::event run(sycl::queue & q, const Args & a) {
    wcopy_run(q, a);
#ifndef FSA_TIMING
    return run_t<0>(q, a);
#else
    switch (a.skip) {
        case 1: return run_t<1>(q, a);
        case 2: return run_t<2>(q, a);
        case 4: return run_t<4>(q, a);
        case 8: return run_t<8>(q, a);
        case 10: return run_t<10>(q, a);
        case 16: return run_t<16>(q, a);
        case 31: return run_t<31>(q, a);
        default: return run_t<0>(q, a);
    }
#endif
}

}  // namespace fsa
