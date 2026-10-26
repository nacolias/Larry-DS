// ggml-sycl/fattn-dsv4-dec.hpp - larry (DeepSeek-V4.1 on Intel Arc): flash attention for the DECODE / verify shape.
//
// Each of n (< 8) queries attends with 64 MQA heads (D 512) to its own key set (its window rows + its top-k compressed
// rows, M <= ~640, K = V) plus one sink logit per head.  The generic TILE kernel took 62 us per layer for one query,
// and the graph around it padded M to 256 and cast K and the mask to f16 (six more kernels).  Here, flash-decoding:
//   1. per (query, KC-key chunk, HG heads): the chunk's keys -> local memory, one sub-group per head: scores (lanes
//      split D; a reduce-scatter leaves keys KPL L .. KPL L + KPL - 1 on lane L), the chunk's max / sum, its O share
//   2. per (query, head), one sub-group: the chunks' parts and the sink combined into O.
// K and the mask may be f32 (no casts in the graph) or f16; any M.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace dsv4dec {

constexpr int DD = 512;  // head dim this kernel is built for

struct Args {
    const float * q = nullptr;   // q[b * q_b + h * q_h + d]
    size_t q_h = 0, q_b = 0;
    const void * k = nullptr;    // K[b * k_b + j * D + d] (= V)
    bool k_f16 = false;
    size_t k_b = 0;
    const void * mask = nullptr; // mask[b * m_b + j]
    bool m_f16 = false;
    size_t m_b = 0;
    const float * sinks = nullptr;  // [NH] or null
    float * o = nullptr;         // o[(b * NH + h) * D + d]
    int D = DD, NH = 0, M = 0, n = 0;
    float scale = 1.0f;
    float * part = nullptr;      // scratch_floats() floats
};

inline int n_chunks(int M, int KC) { return (M + KC - 1) / KC; }
inline size_t scratch_floats(int NH, int M, int n, int KC = 32) { return (size_t) n * n_chunks(M, KC) * NH * (DD + 2); }

// KC keys per chunk (16 or 32), HG heads (sub-groups) per work-group, KT the local copy's type (half / float)
template <int KC, int HG, typename KT>
sycl::event launch_t(sycl::queue & q, const Args & a, sycl::event * first = nullptr) {
    static_assert(KC == 16 || KC == 32, "KC");
    constexpr int KPL = KC / 16;   // keys per lane after the reduce-scatter
    const int nch = n_chunks(a.M, KC);
    const int NH = a.NH, M = a.M, n = a.n;
    constexpr int D = DD, DPL = D / 16, LD = D + (sizeof(KT) == 2 ? 8 : 4);   // dims per lane; local row stride
    const Args A = a;
    sycl::event e1 = q.submit([&](sycl::handler & h) {
        sycl::local_accessor<KT, 1> kl(sycl::range<1>((size_t) KC * LD), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) n, (size_t) nch, (size_t) (NH / HG) * HG * 16),
                                         sycl::range<3>(1, 1, (size_t) HG * 16)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int b = (int) it.get_group(0), c = (int) it.get_group(1), hg = (int) it.get_group(2);
            const int lid = (int) it.get_local_id(2);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int hh = hg * HG + s;
            const int j0 = c * KC;
            const int nk = sycl::min(KC, M - j0);
            // 1. the chunk's keys -> local memory, 4 values per step (unrolled: the loads in flight together)
#pragma unroll 8
            for (int idx = lid * 4; idx < KC * D; idx += HG * 16 * 4) {
                const int row = idx / D, d = idx % D;
                sycl::float4 fv(0.0f, 0.0f, 0.0f, 0.0f);
                if (row < nk) {
                    const size_t g = (size_t) b * A.k_b + (size_t) (j0 + row) * D + d;
                    if (A.k_f16) {
                        const sycl::half4 hv = *reinterpret_cast<const sycl::half4 *>((const sycl::half *) A.k + g);
                        fv = sycl::float4((float) hv.x(), (float) hv.y(), (float) hv.z(), (float) hv.w());
                    } else {
                        fv = *reinterpret_cast<const sycl::float4 *>((const float *) A.k + g);
                    }
                }
                *reinterpret_cast<sycl::vec<KT, 4> *>(&kl[(size_t) row * LD + d]) = fv.template convert<KT>();
            }
            // the head's query in registers (scaled): lane L's dims are 128 g + 8 L + e (g < 4, e < 8)
            float qr[DPL];
            const float * qp = A.q + (size_t) b * A.q_b + (size_t) hh * A.q_h;
#pragma unroll
            for (int g = 0; g < 4; ++g)
#pragma unroll
                for (int e = 0; e < 8; ++e) qr[8 * g + e] = qp[128 * g + 8 * lane + e] * A.scale;
            sycl::group_barrier(it.get_group());
            using v8 = sycl::vec<KT, 8>;
            // 2. scores: per key the lane's partial dot over its dims
            float sc[KC];
#pragma unroll
            for (int j = 0; j < KC; ++j) {
                float acc = 0.0f;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const v8 kv = *reinterpret_cast<const v8 *>(&kl[(size_t) j * LD + 128 * g + 8 * lane]);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc += qr[8 * g + e] * (float) kv[e];
                }
                sc[j] = acc;
            }
            // reduce-scatter over the 16 lanes: lane L ends with the totals of keys KPL L + e
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const int mask = 8 >> st;
                const int half = KC >> (st + 1);
                const bool hi = (lane & mask) != 0;
#pragma unroll
                for (int k = 0; k < half; ++k) {
                    const float send = hi ? sc[k] : sc[k + half];
                    const float keep = hi ? sc[k + half] : sc[k];
                    sc[k] = keep + sycl::permute_group_by_xor(sg, send, mask);
                }
            }
            // 3. mask, the chunk's max and sum (masked / missing keys: weight 0)
            float v[KPL], p[KPL];
            float vm = -1e30f;
#pragma unroll
            for (int e = 0; e < KPL; ++e) {
                const int key = KPL * lane + e;
                float x = -1e30f;
                if (key < nk) {
                    const size_t mi = (size_t) b * A.m_b + (size_t) (j0 + key);
                    const float mv = A.m_f16 ? (float) ((const sycl::half *) A.mask)[mi] : ((const float *) A.mask)[mi];
                    x = sycl::fmax(sc[e] + sycl::fmax(mv, -1e30f), -1e30f);
                }
                v[e] = x;
                vm = sycl::fmax(vm, x);
            }
            const float mx = sycl::reduce_over_group(sg, vm, sycl::maximum<float>());
            float ps = 0.0f;
#pragma unroll
            for (int e = 0; e < KPL; ++e) { p[e] = v[e] > -1e29f ? sycl::exp(v[e] - mx) : 0.0f; ps += p[e]; }
            const float l = sycl::reduce_over_group(sg, ps, sycl::plus<float>());
            // 4. this chunk's O share: the lane's dims as above
            float o[DPL];
#pragma unroll
            for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
#pragma unroll
            for (int j = 0; j < KC; ++j) {
                const float pj = sycl::select_from_group(sg, p[j % KPL], j / KPL);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const v8 kv = *reinterpret_cast<const v8 *>(&kl[(size_t) j * LD + 128 * g + 8 * lane]);
#pragma unroll
                    for (int e = 0; e < 8; ++e) o[8 * g + e] += pj * (float) kv[e];
                }
            }
            float * pp = A.part + (((size_t) b * nch + c) * NH + hh) * (D + 2);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                *reinterpret_cast<sycl::float4 *>(pp + 128 * g + 8 * lane) = sycl::float4(o[8 * g], o[8 * g + 1], o[8 * g + 2], o[8 * g + 3]);
                *reinterpret_cast<sycl::float4 *>(pp + 128 * g + 8 * lane + 4) = sycl::float4(o[8 * g + 4], o[8 * g + 5], o[8 * g + 6], o[8 * g + 7]);
            }
            if (lane == 0) { pp[D] = mx; pp[D + 1] = l; }
        });
    });
    if (first) *first = e1;
    // 5. combine: one sub-group per (query, head); lane c (and c + 16, ...) reads chunk c's max / sum, the weights go
    // round by shuffles, every lane adds up its dims (as above) over the chunks
    constexpr int SGW = 4;
    const size_t rows = (size_t) n * NH;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((rows + SGW - 1) / SGW * SGW * 16), sycl::range<1>(SGW * 16)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t row = (size_t) it.get_group(0) * SGW + sg.get_group_linear_id();   // b * NH + h
        if (row >= rows) return;
        const int lane = (int) sg.get_local_linear_id();
        const int b = (int) (row / NH), hh = (int) (row % NH);
        const float * pb = A.part + ((size_t) b * nch * NH + hh) * (D + 2);
        const size_t cs = (size_t) NH * (D + 2);   // chunk stride
        constexpr int MAXC = 64;                   // chunks: <= 4 per lane (M <= 1024 at KC 16)
        float mc[MAXC / 16], lc[MAXC / 16];
        float gm = A.sinks ? A.sinks[hh] : -1e30f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) {
            const int c = lane + 16 * r;
            mc[r] = c < nch ? pb[c * cs + D] : -1e30f;
            lc[r] = c < nch ? pb[c * cs + D + 1] : 0.0f;
            gm = sycl::fmax(gm, mc[r]);
        }
        gm = sycl::reduce_over_group(sg, gm, sycl::maximum<float>());
        float w[MAXC / 16], ls = 0.0f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) { w[r] = mc[r] > -1e29f ? sycl::exp(mc[r] - gm) : 0.0f; ls += lc[r] * w[r]; }
        float L = sycl::reduce_over_group(sg, ls, sycl::plus<float>());
        if (A.sinks) L += sycl::exp(A.sinks[hh] - gm);
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        float acc[DPL];
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;
        for (int c = 0; c < nch; ++c) {
            const int r = c / 16;
            const float wr = r == 0 ? w[0] : r == 1 ? w[1] : r == 2 ? w[2] : w[3];
            const float wc = sycl::select_from_group(sg, wr, c % 16);
            const float * pc = pb + c * cs;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const sycl::float4 x0 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane);
                const sycl::float4 x1 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane + 4);
                acc[8 * g] += x0.x() * wc; acc[8 * g + 1] += x0.y() * wc; acc[8 * g + 2] += x0.z() * wc; acc[8 * g + 3] += x0.w() * wc;
                acc[8 * g + 4] += x1.x() * wc; acc[8 * g + 5] += x1.y() * wc; acc[8 * g + 6] += x1.z() * wc; acc[8 * g + 7] += x1.w() * wc;
            }
        }
        float * op = A.o + row * D;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane) =
                sycl::float4(acc[8 * g] * inv, acc[8 * g + 1] * inv, acc[8 * g + 2] * inv, acc[8 * g + 3] * inv);
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane + 4) =
                sycl::float4(acc[8 * g + 4] * inv, acc[8 * g + 5] * inv, acc[8 * g + 6] * inv, acc[8 * g + 7] * inv);
        }
    });
}

// ---- the XMX form: per (query, 64-key chunk, 32 heads) work-group, the chunk's keys (f16) and Q^T (f16, scaled) in
// local memory, S^T = K . Q^T and O = P . K as joint_matrix tiles (8 x 16 x 16), the chunk's softmax in between.
// Parts: O [query][chunk][head][D] and (max, sum) [query][chunk][head][2] (16-byte aligned rows for the matrix stores).
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int XKC = 64, XHG = 32;

inline size_t xmx_scratch_floats(int NH, int M, int n) { return (size_t) n * n_chunks(M, XKC) * NH * (DD + 2); }

inline sycl::event launch_xmx(sycl::queue & q, const Args & a, sycl::event * first = nullptr) {
    const int nch = n_chunks(a.M, XKC);
    const int NH = a.NH, M = a.M, n = a.n;
    constexpr int D = DD;
    constexpr int KS = D + 8;     // ks row stride (halves): ks[key][d]
    constexpr int QS = XHG + 8;   // qt row stride (halves): qt[d][head]
    constexpr int SS = XHG + 4;   // ss row stride (floats): ss[key][head]
    constexpr int PS = XKC + 16;  // ps row stride (halves): ps[head][key]
    const Args A = a;
    float * pO = a.part;
    float * pML = a.part + (size_t) n * nch * NH * D;
    sycl::event e1 = q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> ks(sycl::range<1>((size_t) XKC * KS), h);
        sycl::local_accessor<sycl::half, 1> qt(sycl::range<1>((size_t) D * QS), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>((size_t) XKC * SS), h);
        sycl::local_accessor<sycl::half, 1> ps(sycl::range<1>((size_t) XHG * PS), h);
        sycl::local_accessor<float, 1> mk(sycl::range<1>((size_t) XKC), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) n, (size_t) nch, (size_t) (NH / XHG) * 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int b = (int) it.get_group(0), c = (int) it.get_group(1), hg = (int) it.get_group(2);
            const int lid = (int) it.get_local_id(2);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int j0 = c * XKC;
            const int nk = sycl::min(XKC, M - j0);
            // 1. the chunk's keys -> ks (f16; rows past the key count are zero), unrolled
#pragma unroll 8
            for (int idx = lid * 4; idx < XKC * D; idx += 256 * 4) {
                const int row = idx / D, d = idx % D;
                sycl::half4 hv(0.0f, 0.0f, 0.0f, 0.0f);
                if (row < nk) {
                    const size_t g = (size_t) b * A.k_b + (size_t) (j0 + row) * D + d;
                    if (A.k_f16) {
                        hv = *reinterpret_cast<const sycl::half4 *>((const sycl::half *) A.k + g);
                    } else {
                        const sycl::float4 fv = *reinterpret_cast<const sycl::float4 *>((const float *) A.k + g);
                        hv = sycl::half4((sycl::half) fv.x(), (sycl::half) fv.y(), (sycl::half) fv.z(), (sycl::half) fv.w());
                    }
                }
                *reinterpret_cast<sycl::half4 *>(&ks[(size_t) row * KS + d]) = hv;
            }
            // the chunk's mask -> mk (-1e30 for keys past the count)
            if (lid < XKC) {
                float mv = -1e30f;
                if (lid < nk) {
                    const size_t mi = (size_t) b * A.m_b + (size_t) (j0 + lid);
                    mv = sycl::fmax(A.m_f16 ? (float) ((const sycl::half *) A.mask)[mi] : ((const float *) A.mask)[mi], -1e30f);
                }
                mk[lid] = mv;
            }
            // 2. Q^T of this work-group's 32 heads -> qt (scaled f16)
#pragma unroll 8
            for (int idx = lid * 4; idx < XHG * D; idx += 256 * 4) {
                const int hl = idx / D, d = idx % D;
                const sycl::float4 v = *reinterpret_cast<const sycl::float4 *>(A.q + (size_t) b * A.q_b + (size_t) (hg * XHG + hl) * A.q_h + d);
                qt[(size_t) d * QS + hl] = (sycl::half) (v.x() * A.scale);
                qt[(size_t) (d + 1) * QS + hl] = (sycl::half) (v.y() * A.scale);
                qt[(size_t) (d + 2) * QS + hl] = (sycl::half) (v.z() * A.scale);
                qt[(size_t) (d + 3) * QS + hl] = (sycl::half) (v.w() * A.scale);
            }
            sycl::group_barrier(it.get_group());
            auto kp = ks.template get_multi_ptr<sycl::access::decorated::no>();
            // 3. S^T: one 8-key x 16-head tile per sub-group (keys 8 mt.., heads 16 nt..), 32 steps over D
            {
                const int mt = s >> 1, nt = s & 1;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
                jm::joint_matrix_fill(sg, mc, 0.0f);
                auto qp = qt.template get_multi_ptr<sycl::access::decorated::no>();
                for (int k = 0; k < D / 16; ++k) {
                    jm::joint_matrix_load(sg, ma, kp + (size_t) (mt * 8) * KS + k * 16, KS);
                    jm::joint_matrix_load(sg, mb, qp + (size_t) (k * 16) * QS + nt * 16, QS);
                    jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                }
                auto sp = ss.template get_multi_ptr<sycl::access::decorated::no>();
                jm::joint_matrix_store(sg, mc, sp + (size_t) (mt * 8) * SS + nt * 16, SS, jm::layout::row_major);
            }
            sycl::group_barrier(it.get_group());
            // 4. the chunk's softmax per head: sub-groups 0, 1, lane = head; P (f16) -> ps[head][key]
            if (s < 2) {
                const int hl = s * 16 + lane, hh = hg * XHG + hl;
                float xv[XKC];
                float mx = -1e30f;
#pragma unroll
                for (int j = 0; j < XKC; ++j) {
                    xv[j] = sycl::fmax(ss[(size_t) j * SS + hl] + mk[j], -1e30f);
                    mx = sycl::fmax(mx, xv[j]);
                }
                float l = 0.0f;
#pragma unroll
                for (int j = 0; j < XKC; ++j) {
                    const float p = xv[j] > -1e29f ? sycl::exp(xv[j] - mx) : 0.0f;
                    l += p;
                    ps[(size_t) hl * PS + j] = (sycl::half) p;
                }
                float * ml = pML + (((size_t) b * nch + c) * NH + hh) * 2;
                ml[0] = mx;
                ml[1] = l;
            }
            sycl::group_barrier(it.get_group());
            // 5. this chunk's O = P . K: 4 head tiles x 32 dim tiles, 8 per sub-group, 4 steps over the keys
            {
                auto pp = ps.template get_multi_ptr<sycl::access::decorated::no>();
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
                for (int ti = 0; ti < 8; ++ti) {
                    const int tile = s * 8 + ti, ht = tile / 32, dt = tile % 32;
                    jm::joint_matrix_fill(sg, mc, 0.0f);
#pragma unroll
                    for (int k = 0; k < XKC / 16; ++k) {
                        jm::joint_matrix_load(sg, ma, pp + (size_t) (ht * 8) * PS + k * 16, PS);
                        jm::joint_matrix_load(sg, mb, kp + (size_t) (k * 16) * KS + dt * 16, KS);
                        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    float * dst = pO + (((size_t) b * nch + c) * NH + hg * XHG + ht * 8) * D + dt * 16;
                    jm::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                             sycl::access::decorated::no>(dst), D, jm::layout::row_major);
                }
            }
        });
    });
    if (first) *first = e1;
    // 6. combine (as launch_t's, with the parts split into O and (max, sum))
    constexpr int SGW = 4;
    const size_t rows = (size_t) n * NH;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((rows + SGW - 1) / SGW * SGW * 16), sycl::range<1>(SGW * 16)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t row = (size_t) it.get_group(0) * SGW + sg.get_group_linear_id();   // b * NH + h
        if (row >= rows) return;
        const int lane = (int) sg.get_local_linear_id();
        const int b = (int) (row / NH), hh = (int) (row % NH);
        const float * ob = pO + ((size_t) b * nch * NH + hh) * D;
        const float * mb = pML + ((size_t) b * nch * NH + hh) * 2;
        const size_t cs = (size_t) NH * D, ms = (size_t) NH * 2;   // chunk strides
        constexpr int MAXC = 32;                                  // chunks: <= 2 per lane (M <= 2048)
        float mc[MAXC / 16], lc[MAXC / 16];
        float gm = A.sinks ? A.sinks[hh] : -1e30f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) {
            const int c = lane + 16 * r;
            mc[r] = c < nch ? mb[c * ms] : -1e30f;
            lc[r] = c < nch ? mb[c * ms + 1] : 0.0f;
            gm = sycl::fmax(gm, mc[r]);
        }
        gm = sycl::reduce_over_group(sg, gm, sycl::maximum<float>());
        float w[MAXC / 16], ls = 0.0f;
#pragma unroll
        for (int r = 0; r < MAXC / 16; ++r) { w[r] = mc[r] > -1e29f ? sycl::exp(mc[r] - gm) : 0.0f; ls += lc[r] * w[r]; }
        float L = sycl::reduce_over_group(sg, ls, sycl::plus<float>());
        if (A.sinks) L += sycl::exp(A.sinks[hh] - gm);
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        float acc[D / 16];
#pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[i] = 0.0f;
        for (int c = 0; c < nch; ++c) {
            const float wc = sycl::select_from_group(sg, c < 16 ? w[0] : w[1], c % 16);
            const float * pc = ob + c * cs;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const sycl::float4 x0 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane);
                const sycl::float4 x1 = *reinterpret_cast<const sycl::float4 *>(pc + 128 * g + 8 * lane + 4);
                acc[8 * g] += x0.x() * wc; acc[8 * g + 1] += x0.y() * wc; acc[8 * g + 2] += x0.z() * wc; acc[8 * g + 3] += x0.w() * wc;
                acc[8 * g + 4] += x1.x() * wc; acc[8 * g + 5] += x1.y() * wc; acc[8 * g + 6] += x1.z() * wc; acc[8 * g + 7] += x1.w() * wc;
            }
        }
        float * op = A.o + row * D;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane) =
                sycl::float4(acc[8 * g] * inv, acc[8 * g + 1] * inv, acc[8 * g + 2] * inv, acc[8 * g + 3] * inv);
            *reinterpret_cast<sycl::float4 *>(op + 128 * g + 8 * lane + 4) =
                sycl::float4(acc[8 * g + 4] * inv, acc[8 * g + 5] * inv, acc[8 * g + 6] * inv, acc[8 * g + 7] * inv);
        }
    });
}

// the configuration the hook uses
inline sycl::event launch(sycl::queue & q, const Args & a, sycl::event * first = nullptr) {
    return launch_t<32, 32, sycl::half>(q, a, first);
}

}  // namespace dsv4dec
