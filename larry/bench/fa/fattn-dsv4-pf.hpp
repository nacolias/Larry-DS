// ggml-sycl/fattn-dsv4-pf.hpp - larry (DeepSeek-V4.1 on Intel Arc): flash attention for the PROMPT shape, in one kernel.
//
// Every query attends with 64 MQA heads (D 512) to its own key set (its window rows + its top-k compressed rows, M <=
// ~640, K = V) plus one sink logit per head.  fattn-dsv4.cpp's oneMKL path does it in four memory-bound passes - Q to
// f16, S = Q . K^T batched (S f32 written: 84 MB for a 512-query span), the softmax (P f16 written), O = P . K batched:
// ~2.4 ms of a ~7.4 ms dense span on card 0 (2026-10-25 unitrace), for ~43 GFLOP.  Here one work-group a query (all
// 64 heads, 32 sub-groups): K in 32-key chunks through local memory (each chunk read once; the next chunk's loads
// issued before this one's math), S^T = K . Q^T as joint_matrix tiles (8 x 16 x 16), an online softmax per head, and
// O += P . K into accumulators that stay in registers (a sub-group: 8 heads x 128 dims), rescaled per chunk.  The same
// products as the oneMKL path (f16 x f16 -> f32, P rounded to f16); only the softmax's running maximum differs (P is
// rounded relative to the max so far; the f32 accumulators are rescaled exactly).
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstddef>
#include <cstdint>

namespace dsv4pf {

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;

constexpr int D = 512;        // head dim
constexpr int NH = 64;        // heads (one work-group: all of them)
constexpr int KC = 32;        // keys a chunk
constexpr int NSG = 32;       // sub-groups a work-group
constexpr int WG = NSG * 16;  // work-items a work-group
constexpr int QS = NH + 8;    // qt[d][head] row stride (halves)
constexpr int KS = D + 8;     // ks[key][d] row stride (halves)
constexpr int SS = NH + 4;    // ss[key][head] row stride (floats)
constexpr int PS = KC + 16;   // ps[head][key] row stride (halves)
constexpr int KPW = KC * D / WG / 4;   // half4 loads of a K chunk a work-item (8)

struct Args {
    const float * q = nullptr;      // q[b * q_b + h * q_h + d]
    size_t q_h = 0, q_b = 0;
    const sycl::half * k = nullptr; // K[b * k_b + j * D + d] (= V)
    size_t k_b = 0;
    const sycl::half * mask = nullptr;   // mask[b * m_b + j] (0 / -inf)
    size_t m_b = 0;
    const float * sinks = nullptr;  // [NH] or null
    float * o = nullptr;            // o[(b * NH + h) * D + d]
    int M = 0, n = 0;
    float scale = 1.0f;
};

/// the shapes this kernel takes (the caller checks the rest: contiguity, 16-byte alignment)
inline bool shape_ok(int d, int nh, int M) { return d == D && nh == NH && M >= 1; }

inline sycl::event launch(sycl::queue & q, const Args & a) {
    const Args A = a;
    const int nch = (A.M + KC - 1) / KC;
    namespace sxo = sycl::ext::oneapi::experimental;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> qt(sycl::range<1>((size_t) D * QS), h);
        sycl::local_accessor<sycl::half, 1> ks(sycl::range<1>((size_t) KC * KS), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>((size_t) KC * SS), h);
        sycl::local_accessor<sycl::half, 1> ps(sycl::range<1>((size_t) NH * PS), h);
        sycl::local_accessor<float, 1> mk(sycl::range<1>((size_t) KC), h);
        sycl::local_accessor<float, 1> al(sycl::range<1>((size_t) NH), h);
        auto kern = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int b = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const sycl::half * kb = A.k + (size_t) b * A.k_b;
            // 0. Q^T (scaled, f16) -> qt[d][head]
            for (int idx = lid * 4; idx < NH * D; idx += WG * 4) {
                const int hh = idx / D, d = idx % D;
                const sycl::float4 v = *reinterpret_cast<const sycl::float4 *>(A.q + (size_t) b * A.q_b + (size_t) hh * A.q_h + d);
                qt[(size_t) d * QS + hh] = (sycl::half) (v.x() * A.scale);
                qt[(size_t) (d + 1) * QS + hh] = (sycl::half) (v.y() * A.scale);
                qt[(size_t) (d + 2) * QS + hh] = (sycl::half) (v.z() * A.scale);
                qt[(size_t) (d + 3) * QS + hh] = (sycl::half) (v.w() * A.scale);
            }
            // a chunk's keys: KPW half4 a work-item, held in registers until the chunk before it is done
            sycl::half4 pre[KPW];
            auto fetch = [&](int c) {
                const int j0 = c * KC, nk = sycl::min(KC, A.M - j0);
#pragma unroll
                for (int i = 0; i < KPW; ++i) {
                    const int idx = (lid + i * WG) * 4, row = idx / D, d = idx % D;
                    pre[i] = row < nk ? *reinterpret_cast<const sycl::half4 *>(kb + (size_t) (j0 + row) * D + d)
                                      : sycl::half4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            };
            fetch(0);
            // the online softmax of head s * 16 + lane (sub-groups 0..3)
            float m_run = -1e30f, l_run = 0.0f;
            // O: this sub-group's 8 heads (tile row ht) x 8 dim tiles (dt0 ..)
            const int ht = s >> 2, dt0 = (s & 3) * 8;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1, c2, c3, c4, c5, c6, c7;
            jm::joint_matrix_fill(sg, c0, 0.0f); jm::joint_matrix_fill(sg, c1, 0.0f);
            jm::joint_matrix_fill(sg, c2, 0.0f); jm::joint_matrix_fill(sg, c3, 0.0f);
            jm::joint_matrix_fill(sg, c4, 0.0f); jm::joint_matrix_fill(sg, c5, 0.0f);
            jm::joint_matrix_fill(sg, c6, 0.0f); jm::joint_matrix_fill(sg, c7, 0.0f);
            auto kp = ks.template get_multi_ptr<sycl::access::decorated::no>();
            auto qp = qt.template get_multi_ptr<sycl::access::decorated::no>();
            auto sp = ss.template get_multi_ptr<sycl::access::decorated::no>();
            auto pp = ps.template get_multi_ptr<sycl::access::decorated::no>();
            for (int c = 0; c < nch; ++c) {
                const int j0 = c * KC, nk = sycl::min(KC, A.M - j0);
                sycl::group_barrier(it.get_group());   // the previous chunk's O update is done with ks / ps / al
                // 1. this chunk's keys -> ks, its mask -> mk; then the next chunk's loads go out
#pragma unroll
                for (int i = 0; i < KPW; ++i) {
                    const int idx = (lid + i * WG) * 4, row = idx / D, d = idx % D;
                    *reinterpret_cast<sycl::half4 *>(&ks[(size_t) row * KS + d]) = pre[i];
                }
                if (lid < KC) mk[lid] = lid < nk ? sycl::fmax((float) A.mask[(size_t) b * A.m_b + (size_t) (j0 + lid)], -1e30f) : -1e30f;
                sycl::group_barrier(it.get_group());
                if (c + 1 < nch) fetch(c + 1);
                // 2. S^T [32 keys x 64 heads]: 4 x 4 tiles, one a sub-group (sub-groups 0..15), 32 steps over D
                if (s < 16) {
                    const int mt = s >> 2, nt = s & 3;
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
                    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
                    jm::joint_matrix_fill(sg, mc, 0.0f);
                    for (int k = 0; k < D / 16; ++k) {
                        jm::joint_matrix_load(sg, ma, kp + (size_t) (mt * 8) * KS + k * 16, KS);
                        jm::joint_matrix_load(sg, mb, qp + (size_t) (k * 16) * QS + nt * 16, QS);
                        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    }
                    jm::joint_matrix_store(sg, mc, sp + (size_t) (mt * 8) * SS + nt * 16, SS, jm::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // 3. the softmax step per head (sub-groups 0..3, lane = head): P (f16) -> ps[head][key], the rescale -> al
                if (s < 4) {
                    const int hh = s * 16 + lane;
                    float mc = -1e30f;
                    for (int j = 0; j < KC; ++j) mc = sycl::fmax(mc, sycl::fmax(ss[(size_t) j * SS + hh] + mk[j], -1e30f));
                    const float m_new = sycl::fmax(m_run, mc);
                    const float alpha = sycl::exp(m_run - m_new);   // 1 while nothing counted yet (-1e30 - -1e30)
                    float lsum = 0.0f;
                    for (int j = 0; j < KC; ++j) {
                        const float x = sycl::fmax(ss[(size_t) j * SS + hh] + mk[j], -1e30f);
                        const float p = x > -1e29f ? sycl::exp(x - m_new) : 0.0f;
                        lsum += p;
                        ps[(size_t) hh * PS + j] = (sycl::half) p;
                    }
                    l_run = l_run * alpha + lsum;
                    m_run = m_new;
                    al[hh] = alpha;
                }
                sycl::group_barrier(it.get_group());
                // 4. O = O * alpha + P . K (two 16-key steps; P's 8 head rows loaded once a step)
                {
                    auto rescale = [&](jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> & cm) {
                        jmi::joint_matrix_apply(sg, cm, [=](float & x, size_t row, size_t) { x *= al[(size_t) (ht * 8) + row]; });
                    };
                    rescale(c0); rescale(c1); rescale(c2); rescale(c3); rescale(c4); rescale(c5); rescale(c6); rescale(c7);
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> ma;
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> mb;
#pragma unroll
                    for (int k = 0; k < KC / 16; ++k) {
                        jm::joint_matrix_load(sg, ma, pp + (size_t) (ht * 8) * PS + k * 16, PS);
#define DSV4PF_PV(CM, TI)                                                                              \
                        jm::joint_matrix_load(sg, mb, kp + (size_t) (k * 16) * KS + (dt0 + TI) * 16, KS); \
                        jm::joint_matrix_mad(sg, CM, ma, mb, CM);
                        DSV4PF_PV(c0, 0) DSV4PF_PV(c1, 1) DSV4PF_PV(c2, 2) DSV4PF_PV(c3, 3)
                        DSV4PF_PV(c4, 4) DSV4PF_PV(c5, 5) DSV4PF_PV(c6, 6) DSV4PF_PV(c7, 7)
#undef DSV4PF_PV
                    }
                }
            }
            // 5. the sink and 1 / sum per head, then O out
            sycl::group_barrier(it.get_group());
            if (s < 4) {
                const int hh = s * 16 + lane;
                const float sk = A.sinks ? A.sinks[hh] : -1e30f;
                const float m_f = sycl::fmax(m_run, sk);
                const float sc = m_run > -1e29f ? sycl::exp(m_run - m_f) : 0.0f;
                const float L = l_run * sc + (A.sinks ? sycl::exp(sk - m_f) : 0.0f);
                al[hh] = L > 0.0f ? sc / L : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            float * ob = A.o + ((size_t) b * NH + (size_t) (ht * 8)) * D;
            auto out = [&](jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> & cm, int ti) {
                jmi::joint_matrix_apply(sg, cm, [=](float & x, size_t row, size_t) { x *= al[(size_t) (ht * 8) + row]; });
                jm::joint_matrix_store(sg, cm, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                         sycl::access::decorated::no>(ob + (dt0 + ti) * 16),
                                       D, jm::layout::row_major);
            };
            out(c0, 0); out(c1, 1); out(c2, 2); out(c3, 3); out(c4, 4); out(c5, 5); out(c6, 6); out(c7, 7);
        };
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)),
                       sxo::properties{sycl::ext::intel::experimental::grf_size<256>}, kern);
    });
}

}  // namespace dsv4pf
