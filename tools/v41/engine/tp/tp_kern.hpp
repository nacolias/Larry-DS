// tools/v41/engine/tp/tp_kern.hpp - larry (DeepSeek-V4.1 on N Intel Arc B70s): the kernels of the tensor-parallel
// decode core (engine1, tp_core.cpp).  Pointer-based, no ggml.  Most are ggml-sycl's fused decode kernels taken over
// (dsv41-dec.hpp is included as is, in its own namespace; the hc / router / swiglu kernels of dsv41-fused.cpp and the
// radix top-k of topk-radix.cpp are copied here with their arithmetic unchanged); new here: the cross-card sum's
// consumer side (hc_fab reads the N cards' partials and adds them in rank order), the sync kernels, the compressor
// chain of a kv source layer, the Engram add, the state taps.
//
// Shapes are DeepSeek-V4.1-Flash's (the geometry is checked by the host): D = 5120, HC = 4, DH = 512, 24 hc mixes.
#pragma once

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

// the TP library's own copies of ggml-sycl's decode kernels and mat-vecs (the engine links ggml-sycl statically: other
// namespaces keep the two copies' host functions and SYCL kernel names apart)
#define dsv41dec tpdec
#include "dsv41-dec.hpp"
#define fastmv tpfmv
#include "fastmv.hpp"
namespace tpfmv {
// tp_fastmv.cpp: a Q6K_R8 mat-vec in a named form (0 fastmv's dispatch, 1-4 dp4a rows 2 / 4 x normal / 256 registers,
// 5-8 XMX with K split 1 / 2 / 4 / 8: xT / xdT in quant_xT's layout); false when the form cannot take the shape
bool launch_form(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT, int form);
}  // namespace tpfmv
#include "xe_kernels.hpp"   // the xe tier's planar int8 expert kernels (libds41_xe) and their job format

namespace tpk {

constexpr int HC = 4, NMIX = 24;
constexpr int MAXS = 256;          // cross-card sync points a pass (2 a layer + the Engram gathers)
#ifndef TP_WAIT_FENCE
#define TP_WAIT_FENCE 1
#endif
constexpr int64_t SPIN_MAX = (int64_t) 1 << 26;        // device-memory polls before a wait gives up (err[0]): ~1 s
constexpr int64_t SPIN_MAX_HOST = (int64_t) 1 << 22;   // host-memory polls (~1.3 us each over PCIe): ~5 s (err[1])

// ---- a sum of np partials: p[k * stride + t * ts + i], added in k order (every card the same association)
struct PSum {
    const float * p = nullptr;
    int64_t stride = 0, ts = 0;
    int np = 0;
};

// ================================================================== init: x = the embedding in every hc stream,
// pre = one-hot (model.py make_identity_pre_mix)
inline sycl::event init_state(sycl::queue & q, const float * emb, int n, int D, float * x, float * pre) {
    const int nq = D / 4;
    return q.parallel_for(sycl::range<1>((size_t) n * nq), [=](sycl::id<1> id) {
        const int t = (int) (id[0] / nq), i = (int) (id[0] % nq) * 4;
        const sycl::float4 v = *(const sycl::float4 *) (emb + (size_t) t * D + i);
#pragma unroll
        for (int h = 0; h < HC; ++h) *(sycl::float4 *) (x + ((size_t) t * HC + h) * D + i) = v;
        if (i == 0)
#pragma unroll
            for (int h = 0; h < HC; ++h) pre[t * HC + h] = h == 0 ? 1.0f : 0.0f;
    });
}

// ================================================================== the hc step of a sub-layer (dsv41-fused.cpp's
// hc_fa_part + hc_fab_sycl: HC_FA + HC_FB as one kernel).  POST: h = hc_post(y, x, post, comb) with y the SUM of the
// cards' partials (the attention output / the FFN output); else h = x.
template <bool W16, bool POST>
inline void hc_fa_part(const sycl::sub_group & sg, int lane, int64_t t, int64_t ch, int rg, const float * x, int64_t xs1,
                       int64_t xs2, const void * fn, int64_t fs1, const float * pre, int64_t ps1, const PSum & y,
                       const float * post, int64_t pos1, const float * comb, int64_t cs1, int64_t cs2, float * out, int64_t D,
                       int64_t n, float * hdst) {
    constexpr int NR = NMIX, NRG = 4, RPG = NR / NRG, IPL = 8, SG = 16, NV = 32;
    const int64_t nch = D / (SG * IPL);
    float * oh = out;
    float * oz = out + n * HC * D;
    float * op = oz + n * D;
    const int64_t i0 = ch * (SG * IPL) + lane * IPL;
    float h[HC][IPL];
    const float * xt = x + t * xs2 + i0;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        const sycl::float4 va = *(const sycl::float4 *) (xt + s * xs1), vb = *(const sycl::float4 *) (xt + s * xs1 + 4);
        h[s][0] = va.x(); h[s][1] = va.y(); h[s][2] = va.z(); h[s][3] = va.w();
        h[s][4] = vb.x(); h[s][5] = vb.y(); h[s][6] = vb.z(); h[s][7] = vb.w();
    }
    if constexpr (POST) {
        float xr[HC][IPL];
#pragma unroll
        for (int s = 0; s < HC; ++s)
#pragma unroll
            for (int j = 0; j < IPL; ++j) xr[s][j] = h[s][j];
        const float * yp = y.p + t * y.ts + i0;
        sycl::float4 ya = *(const sycl::float4 *) yp, yb = *(const sycl::float4 *) (yp + 4);
        for (int k = 1; k < y.np; ++k) {
            ya += *(const sycl::float4 *) (yp + k * y.stride);
            yb += *(const sycl::float4 *) (yp + k * y.stride + 4);
        }
        const float yv[IPL] = { ya.x(), ya.y(), ya.z(), ya.w(), yb.x(), yb.y(), yb.z(), yb.w() };
        const float * pp = post + t * pos1;
        const float * cc = comb + t * cs2;
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            const float pd = pp[d];
            float cd[HC];
#pragma unroll
            for (int s = 0; s < HC; ++s) cd[s] = cc[d + s * cs1];
#pragma unroll
            for (int j = 0; j < IPL; ++j) {
                float v = yv[j] * pd;
#pragma unroll
                for (int s = 0; s < HC; ++s) v += xr[s][j] * cd[s];
                h[d][j] = v;
            }
        }
    }
    const float * pw = pre + t * ps1;
    const float p0 = pw[0], p1 = pw[1], p2 = pw[2], p3 = pw[3];
    float z[IPL];
#pragma unroll
    for (int j = 0; j < IPL; ++j) {
        float v = h[0][j] * p0;
        v += h[1][j] * p1;
        v += h[2][j] * p2;
        v += h[3][j] * p3;
        z[j] = v;
    }
    float * P = op + (t * nch + ch) * NV;
    if (rg == 0) {
        if constexpr (POST) {
#pragma unroll
            for (int d = 0; d < HC; ++d) {
                float * o = oh + t * HC * D + d * D + i0;
                *(sycl::float4 *) o = sycl::float4(h[d][0], h[d][1], h[d][2], h[d][3]);
                *(sycl::float4 *) (o + 4) = sycl::float4(h[d][4], h[d][5], h[d][6], h[d][7]);
                if (hdst) {
                    float * o2 = hdst + t * HC * D + d * D + i0;
                    *(sycl::float4 *) o2 = sycl::float4(h[d][0], h[d][1], h[d][2], h[d][3]);
                    *(sycl::float4 *) (o2 + 4) = sycl::float4(h[d][4], h[d][5], h[d][6], h[d][7]);
                }
            }
        }
        *(sycl::float4 *) (oz + t * D + i0) = sycl::float4(z[0], z[1], z[2], z[3]);
        *(sycl::float4 *) (oz + t * D + i0 + 4) = sycl::float4(z[4], z[5], z[6], z[7]);
        float sh = 0.0f, sz = 0.0f;
#pragma unroll
        for (int j = 0; j < IPL; ++j) {
#pragma unroll
            for (int d = 0; d < HC; ++d) sh += h[d][j] * h[d][j];
            sz += z[j] * z[j];
        }
        sh = sycl::reduce_over_group(sg, sh, sycl::plus<float>());
        sz = sycl::reduce_over_group(sg, sz, sycl::plus<float>());
        if (lane == 0) { P[NR] = sh; P[NR + 1] = sz; }
    }
    float acc[RPG];
#pragma unroll
    for (int rr = 0; rr < RPG; ++rr) {
        const int64_t r = rg * RPG + rr;
        float a = 0.0f;
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            float wv[IPL];
            if constexpr (W16) {
                const sycl::vec<sycl::half, 8> hv = *(const sycl::vec<sycl::half, 8> *) ((const sycl::half *) fn + r * fs1 + d * D + i0);
#pragma unroll
                for (int j = 0; j < IPL; ++j) wv[j] = (float) hv[j];
            } else {
                const float * wr = (const float *) fn + r * fs1 + d * D + i0;
                const sycl::float4 wa = *(const sycl::float4 *) wr, wb = *(const sycl::float4 *) (wr + 4);
                wv[0] = wa.x(); wv[1] = wa.y(); wv[2] = wa.z(); wv[3] = wa.w();
                wv[4] = wb.x(); wv[5] = wb.y(); wv[6] = wb.z(); wv[7] = wb.w();
            }
#pragma unroll
            for (int j = 0; j < IPL; ++j) a += wv[j] * h[d][j];
        }
        acc[rr] = a;
    }
#pragma unroll
    for (int rr = 0; rr < RPG; ++rr) acc[rr] = sycl::reduce_over_group(sg, acc[rr], sycl::plus<float>());
    if (lane == 0) {
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) P[rg * RPG + rr] = acc[rr];
    }
}

// the hc step's outputs: xn = rms_norm(z) * w (+ its q8: fastmv's int8 rows), pre / post (sigmoid mixes), comb
// (Sinkhorn); h itself (POST) in `out` (and hdst)
struct HcOut {
    const float * sc = nullptr, * bs = nullptr, * pw = nullptr;   // hc scale [3], hc base [24], the norm weight [D]
    float eps = 0.0f, hc_eps = 0.0f;
    int n_iter = 0;
    float * xn = nullptr, * pre = nullptr, * post = nullptr, * comb = nullptr;
    int8_t * xq = nullptr;   // [n][D] (null: no q8)
    float * xd = nullptr;    // [n][D / 32]
    sycl::half * xT = nullptr;   // optional: the same q8 in the XMX mat-vecs' layout [D][8] (fp16 integers) ...
    float * xdT = nullptr;       // ... and its scales [D / 32][8]
};

struct HcIn {
    const float * x = nullptr; int64_t xs1 = 0, xs2 = 0;   // the residual [n][HC][D] (stream stride, token stride)
    const void * fn = nullptr; int64_t fs1 = 0; bool w16 = true;   // the mixes projection [24][HC * D]
    const float * pre = nullptr; int64_t ps1 = 0;           // the previous sub-layer's pre-mix [n][HC]
    PSum y;                                                 // POST: the sub-layer output (summed over the cards)
    const float * post = nullptr; int64_t pos1 = 0;
    const float * comb = nullptr; int64_t cs1 = 0, cs2 = 0;
    float * out = nullptr;      // scratch: h [n][HC][D] | z [n][D] | chunk partials [n][D / 128][32]
    float * hdst = nullptr;     // POST: h also here (the residual the next sub-layer adds to)
    uint32_t * cnt = nullptr;   // [>= n] zeroed once; every launch leaves it zero
    int64_t D = 0, n = 0;
};

inline size_t hc_scratch_floats(int64_t D, int64_t n) { return (size_t) (n * HC * D + n * D + n * (D / 128) * 32); }

template <bool W16, bool POST>
inline sycl::event hc_fab_t(sycl::queue & q, const HcIn & I, const HcOut & F) {
    constexpr int NR = NMIX, NRG = 4, IPL = 8, SG = 16, WG = 256, NV = 32, NCG = WG / NV;
    const int64_t D = I.D, n = I.n;
    const int64_t nch = D / (SG * IPL), nwgt = nch * NRG / (WG / SG), nb = (D + 4 * WG - 1) / (4 * WG);
    const float * fz = I.out + n * HC * D;
    const float * fp = fz + n * D;
    const HcIn A = I;
    const HcOut O = F;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(WG), h), tot(sycl::range<1>(NV), h);
        sycl::local_accessor<int, 1> lastf(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * nwgt * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g = (int64_t) it.get_group(0), t = g / nwgt, wt = g % nwgt;
            const int l = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t sgg = wt * (WG / SG) + (int64_t) sg.get_group_linear_id();
            hc_fa_part<W16, POST>(sg, (int) sg.get_local_linear_id(), t, sgg / NRG, (int) (sgg % NRG), A.x, A.xs1, A.xs2, A.fn,
                                  A.fs1, A.pre, A.ps1, A.y, A.post, A.pos1, A.comb, A.cs1, A.cs2, A.out, D, n, A.hdst);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            sycl::group_barrier(it.get_group());
            if (l == 0) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> a(A.cnt[t]);
                lastf[0] = a.fetch_add(1u) == (uint32_t) (nwgt - 1) ? 1 : 0;
            }
            sycl::group_barrier(it.get_group());
            if (!lastf[0]) return;
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            const int v = l % NV, cg = l / NV;
            float s = 0.0f;
            if (v < NR + 2)
                for (int64_t c = cg; c < nch; c += NCG) s += fp[(t * nch + c) * NV + v];
            part[l] = s;
            it.barrier(sycl::access::fence_space::local_space);
            if (l < NV) {
                float a = 0.0f;
#pragma unroll
                for (int k = 0; k < NCG; ++k) a += part[k * NV + l];
                tot[l] = a;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const float rz = sycl::rsqrt(tot[NR + 1] / (float) D + O.eps);
            for (int64_t blk = 0; blk < nb; ++blk) {
                const int64_t i = blk * 4 * WG + (int64_t) l * 4;
                sycl::float4 xv(0.0f, 0.0f, 0.0f, 0.0f);
                if (i < D) {
                    const sycl::float4 zv = *(const sycl::float4 *) (fz + t * D + i);
                    const sycl::float4 wv = *(const sycl::float4 *) (O.pw + i);
                    xv = zv * rz * wv;
                    *(sycl::float4 *) (O.xn + t * D + i) = xv;
                }
                if (O.xq) {
                    float am = sycl::fmax(sycl::fmax(sycl::fabs(xv.x()), sycl::fabs(xv.y())), sycl::fmax(sycl::fabs(xv.z()), sycl::fabs(xv.w())));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 1));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 2));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 4));
                    if (i < D) {
                        const float id = tpdec::q8_inv(am);
                        *(uint32_t *) (O.xq + t * D + i) = tpdec::q8_pack4(xv.x(), xv.y(), xv.z(), xv.w(), id);
                        if ((l & 7) == 0) O.xd[t * (D / 32) + i / 32] = am / 127.0f;
                        if (O.xT) {
                            O.xT[(i + 0) * 8 + t] = (sycl::half) sycl::rint(xv.x() * id);
                            O.xT[(i + 1) * 8 + t] = (sycl::half) sycl::rint(xv.y() * id);
                            O.xT[(i + 2) * 8 + t] = (sycl::half) sycl::rint(xv.z() * id);
                            O.xT[(i + 3) * 8 + t] = (sycl::half) sycl::rint(xv.w() * id);
                            if ((l & 7) == 0) O.xdT[(i / 32) * 8 + t] = am / 127.0f;
                        }
                    }
                }
            }
            if (l < 16) {   // the whole first sub-group: the mixes and the Sinkhorn
                const float rh = sycl::rsqrt(tot[NR] / (float) (HC * D) + O.eps);
                if (l < 2 * HC) {
                    const bool is_post = l >= HC;
                    const float m = tot[l] * rh * O.sc[is_post ? 1 : 0] + O.bs[l];
                    const float sgm = 1.0f / (1.0f + sycl::exp(-m));
                    if (is_post) O.post[t * HC + (l - HC)] = 2.0f * sgm;
                    else O.pre[t * HC + l] = sgm + O.hc_eps;
                }
                float c = tot[2 * HC + l] * rh * O.sc[2] + O.bs[2 * HC + l];
                float mx = sycl::fmax(c, sycl::permute_group_by_xor(sg, c, 1));
                mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                c = sycl::exp(c - mx);
                float sm = c + sycl::permute_group_by_xor(sg, c, 1);
                sm += sycl::permute_group_by_xor(sg, sm, 2);
                c = c * (1.0f / sm) + O.hc_eps;
                float cs = c + sycl::permute_group_by_xor(sg, c, 4);
                cs += sycl::permute_group_by_xor(sg, cs, 8);
                c *= 1.0f / (cs + O.hc_eps);
                for (int k = 1; k < O.n_iter; ++k) {
                    float rs = c + sycl::permute_group_by_xor(sg, c, 1);
                    rs += sycl::permute_group_by_xor(sg, rs, 2);
                    c *= 1.0f / (rs + O.hc_eps);
                    cs = c + sycl::permute_group_by_xor(sg, c, 4);
                    cs += sycl::permute_group_by_xor(sg, cs, 8);
                    c *= 1.0f / (cs + O.hc_eps);
                }
                O.comb[t * HC * HC + l] = c;
                if (l == 0) A.cnt[t] = 0u;   // ready for the next launch
            }
        });
    });
}

inline sycl::event hc_fab(sycl::queue & q, const HcIn & I, const HcOut & F, bool post) {
    if (post) return I.w16 ? hc_fab_t<true, true>(q, I, F) : hc_fab_t<false, true>(q, I, F);
    return I.w16 ? hc_fab_t<true, false>(q, I, F) : hc_fab_t<false, false>(q, I, F);
}

// ================================================================== the router (dsv41-fused.cpp's DSV41_ROUTER):
// sqrt(softplus(logits)), top k of score + bias + rbias (ties: lower index), weights from the unbiased scores
struct RouterArgs {
    const float * lg = nullptr;     // [n][ne] the router mat-vec's output
    const float * bias = nullptr;   // [ne] exp_probs_b
    const float * rbias = nullptr;  // [ne] the selection-only bias (zeros: the model's routing)
    int ne = 0, k = 0, n = 0;
    bool norm = false;
    float scale = 1.0f;
    int32_t * ids = nullptr;        // [n][k]
    float * wts = nullptr;          // [n][k]
};

inline sycl::event router(sycl::queue & q, const RouterArgs & a) {
    constexpr int WG = 128, NSG = WG / 16, VPL = 8, KMAX = 8;
    const RouterArgs A = a;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> cv(sycl::range<1>(NSG * KMAX), h);
        sycl::local_accessor<int, 1> ci(sycl::range<1>(NSG * KMAX), h);
        sycl::local_accessor<float, 1> cw(sycl::range<1>(NSG * KMAX), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), s = (int) sg.get_group_linear_id();
            const int lid = (int) it.get_local_id(0);
            const int64_t t = (int64_t) it.get_group(0);
            const float * x = A.lg + t * A.ne;
            const int k = A.k;
            float sc[VPL], sel[VPL];
#pragma unroll
            for (int v = 0; v < VPL; ++v) {
                const int i = lid + WG * v;
                if (i < A.ne) {
                    const float ax = sycl::fabs(x[i]);
                    sc[v] = sycl::sqrt(sycl::fmax(x[i], 0.0f) + sycl::log1p(sycl::exp(-ax)));
                    sel[v] = sc[v] + A.bias[i] + A.rbias[i];
                } else {
                    sc[v] = 0.0f;
                    sel[v] = -3.0e38f;
                }
            }
            for (int r = 0; r < k; ++r) {
                float bv = -3.0e38f, bw = 0.0f;
                int bi = 1 << 30, bvv = 0;
#pragma unroll
                for (int v = 0; v < VPL; ++v)
                    if (sel[v] > bv) { bv = sel[v]; bi = lid + WG * v; bw = sc[v]; bvv = v; }
                const float gv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                const int gi = sycl::reduce_over_group(sg, bv == gv ? bi : (1 << 30), sycl::minimum<int>());
                if (bi == gi && bv == gv) {
#pragma unroll
                    for (int v = 0; v < VPL; ++v) if (v == bvv) sel[v] = -3.0e38f;
                    cv[s * KMAX + r] = gv;
                    ci[s * KMAX + r] = gi;
                    cw[s * KMAX + r] = bw;
                }
            }
            sycl::group_barrier(it.get_group());
            if (s != 0) return;
            constexpr int CPL = NSG * KMAX / 16;
            float mv[CPL], mw[CPL];
            int mi[CPL];
#pragma unroll
            for (int u = 0; u < CPL; ++u) {
                const int c = lane + 16 * u, sgi = c / KMAX, rr = c % KMAX;
                const bool ok = rr < k;
                mv[u] = ok ? cv[sgi * KMAX + rr] : -3.0e38f;
                mi[u] = ok ? ci[sgi * KMAX + rr] : (1 << 30);
                mw[u] = ok ? cw[sgi * KMAX + rr] : 0.0f;
            }
            float w_my = 0.0f, sum = 0.0f;
            int id_my = -1;
            for (int r = 0; r < k; ++r) {
                float bv = -3.0e38f, bw = 0.0f;
                int bi = 1 << 30, bu = 0;
#pragma unroll
                for (int u = 0; u < CPL; ++u)
                    if (mv[u] > bv || (mv[u] == bv && mi[u] < bi)) { bv = mv[u]; bi = mi[u]; bw = mw[u]; bu = u; }
                const float gv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                const int gi = sycl::reduce_over_group(sg, bv == gv ? bi : (1 << 30), sycl::minimum<int>());
                float w = 0.0f;
                if (bi == gi && bv == gv) {
#pragma unroll
                    for (int u = 0; u < CPL; ++u) if (u == bu) mv[u] = -3.0e38f;
                    w = bw;
                }
                w = sycl::reduce_over_group(sg, w, sycl::plus<float>());
                sum += w;
                if (lane == r) { w_my = w; id_my = gi; }
            }
            if (lane < k) {
                const float wv = (A.norm ? w_my / (sum + 1e-20f) : w_my) * A.scale;
                A.ids[t * k + lane] = id_my;
                A.wts[t * k + lane] = wv;
            }
        });
    });
}

// ================================================================== the shared expert's SwiGLU (dsv41-fused.cpp's
// swiglu_q8): y [n][2N] = gate rows | up rows -> h [n][N] = silu(min(g, L)) * clamp(u, -L, L) and h's q8
inline sycl::event swiglu_q8(sycl::queue & q, const float * y, int64_t N, int64_t n, float lim, float * hout, int8_t * hq,
                             float * hd, sycl::half * hT = nullptr, float * hdT = nullptr) {   // hT / hdT: the XMX layout too
    const int64_t nblk = N / 32;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * nblk * 16)), sycl::range<1>(16)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t g = (int64_t) it.get_group(0), t = g / nblk, bk = g % nblk;
        const int lane = (int) it.get_local_id(0);
        const int64_t i = bk * 32 + 2 * lane;
        float o[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float gt = y[t * 2 * N + i + j], u = y[t * 2 * N + N + i + j];
            if (lim > 0.0f) {
                gt = gt > lim ? lim : gt;
                u = u < -lim ? -lim : (u > lim ? lim : u);
            }
            o[j] = gt / (1.0f + sycl::exp(-gt)) * u;
        }
        *(sycl::float2 *) (hout + t * N + i) = sycl::float2(o[0], o[1]);
        const float am = sycl::reduce_over_group(it.get_sub_group(), sycl::fmax(sycl::fabs(o[0]), sycl::fabs(o[1])), sycl::maximum<float>());
        const float id = tpdec::q8_inv(am);
        *(uint16_t *) (hq + t * N + i) = (uint16_t) (((int) sycl::rint(o[0] * id) & 0xFF) | (((int) sycl::rint(o[1] * id) & 0xFF) << 8));
        if (lane == 0) hd[t * nblk + bk] = am / 127.0f;
        if (hT) {
            hT[i * 8 + t] = (sycl::half) sycl::rint(o[0] * id);
            hT[(i + 1) * 8 + t] = (sycl::half) sycl::rint(o[1] * id);
            if (lane == 0) hdT[bk * 8 + t] = am / 127.0f;
        }
    });
}

// ================================================================== the head's input: z = hc_pre(x, pre) (the
// composite's arithmetic: h0 p0 + h1 p1 + h2 p2 + h3 p3 in that order); rms_norm * w + q8 follow (tpdec::rms_q8)
inline sycl::event hc_pre(sycl::queue & q, const float * x, const float * pre, int n, int D, float * z) {
    const int nq = D / 4;
    return q.parallel_for(sycl::range<1>((size_t) n * nq), [=](sycl::id<1> id) {
        const int t = (int) (id[0] / nq), i = (int) (id[0] % nq) * 4;
        const float * xt = x + (size_t) t * HC * D + i;
        const float * p = pre + t * HC;
        sycl::float4 v = *(const sycl::float4 *) xt * p[0];
        v += *(const sycl::float4 *) (xt + D) * p[1];
        v += *(const sycl::float4 *) (xt + 2 * D) * p[2];
        v += *(const sycl::float4 *) (xt + 3 * D) * p[3];
        *(sycl::float4 *) (z + (size_t) t * D + i) = v;
    });
}

// DSpark's main_hidden tap: out[t * ostride + i] = the HC-mean of x (ds41_generate's sum of x_h / HC, h in order)
inline sycl::event hc_mean(sycl::queue & q, const float * x, int n, int D, float * out, int64_t ostride) {
    return q.parallel_for(sycl::range<1>((size_t) n * D), [=](sycl::id<1> id) {
        const int t = (int) (id[0] / D), i = (int) (id[0] % D);
        const float * xt = x + (size_t) t * HC * D + i;
        float a = 0.0f;
#pragma unroll
        for (int h = 0; h < HC; ++h) a += xt[(size_t) h * D] / (float) HC;
        out[(size_t) t * ostride + i] = a;
    });
}

// ================================================================== cross-card sync (graph mode: no events across
// cards).  Sync point s of pass p carries p * 4096 + s; a card copies seq[s] into each waiting peer's flag slot right
// after the data copies (same in-order queue: the data lands first) and the peer spins, bounded, on its own memory.
inline sycl::event seq_init(sycl::queue & q, const int32_t * pid, int32_t * seq) {
    return q.parallel_for(sycl::range<1>(MAXS), [=](sycl::id<1> id) { seq[id[0]] = *pid * 4096 + (int32_t) id[0]; });
}

// wait until flags[k * MAXS + s] == pid * 4096 + s for every peer k in `mask` (bit k); err[0] counts give-ups
inline sycl::event wait_peers(sycl::queue & q, const int32_t * flags, const int32_t * pid, int s, uint32_t mask, int32_t * err) {
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)), [=](sycl::nd_item<1> it) {
        const int k = (int) it.get_local_id(0);
        if (!((mask >> k) & 1u)) return;
        if (err[0]) return;   // an earlier wait of the pass gave up: the rest run through (bounded hang, wrong output)
        const int32_t want = *pid * 4096 + s;
        sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>
            f(const_cast<int32_t &>(flags[k * MAXS + s]));
        int64_t i = 0;
        while (f.load(sycl::memory_order::acquire) != want && ++i < SPIN_MAX) {}
        if (i >= SPIN_MAX) {
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(err[0]).fetch_add(1);
        }
        // the peers' data landed behind their flags straight in memory: drop this card's cached copies of it
        if (TP_WAIT_FENCE) sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        // err[2]: the longest wait (polls, saturated), err[3]: waits that polled at all
        const int32_t ii = i > 0x7fffffff ? 0x7fffffff : (int32_t) i;
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(err[2]).fetch_max(ii);
        if (i) sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(err[3]).fetch_add(1);
    });
}

// STRATA_TP_PUSH=1: a card's partial into slot `rank` of every peer by P2P stores (one kernel), then the peers' flags
// (a second kernel, in order behind the data: posted writes on the same path) - instead of 2 (N - 1) copy-engine copies
struct Peers { float * dst[8]; int32_t * flag[8]; int np = 0; };
inline sycl::event push_data(sycl::queue & q, const float * src, size_t nf4, Peers P) {
    return q.parallel_for(sycl::range<1>(nf4), [=](sycl::id<1> i) {
        const sycl::float4 v = ((const sycl::float4 *) src)[i];
        for (int k = 0; k < P.np; ++k) ((sycl::float4 *) P.dst[k])[i] = v;
    });
}
inline sycl::event push_flags(sycl::queue & q, const int32_t * seq_s, Peers P) {
    return q.single_task([=] {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        const int32_t v = *seq_s;
        for (int k = 0; k < P.np; ++k)
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(*P.flag[k]).store(v);
    });
}

// a host gate: wait until the pinned host word *flag == want(pid) = pid * 4096 + s.  Each poll is a system-scope acquire
// FENCE then a volatile load: seen ~3 us after the host's write, eager or in a replayed graph, on every card and pinned
// memory kind (engine1/bench/hs_probe2) - a spin of system-scope atomic acquire LOADS mostly never sees it.
// `need` (device, optional): wait only if *need != 0.
inline sycl::event wait_host(sycl::queue & q, const int32_t * flag, const int32_t * pid, int s, const int32_t * need, int32_t * err) {
    return q.single_task([=] {
        if ((need && *need == 0) || err[1]) return;
        const int32_t want = *pid * 4096 + s;
        int64_t i = 0;
        for (; i < SPIN_MAX_HOST; ++i) {
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            if (*(const volatile int32_t *) flag == want) break;
        }
        if (i >= SPIN_MAX_HOST) {
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(err[1]).fetch_add(1);
        }
    });
}

// ================================================================== the Engram add (model.py Engram.forward, the
// composite of ds41_dense build_attn): kv [n][(HC + 1) D] = HC keys | the value; per stream h:
// dot = sum_d rms(x_h)_d wqk_hd rms(key_h)_d / sqrt(D), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))),
// x_h += gate * value.  One 256-lane work-group per (token, stream).
inline sycl::event engram_add(sycl::queue & q, float * x, const float * kv, const float * wqk, int n, int D, float eps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(3 * WG / 16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n * HC * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int g = (int) it.get_group(0), t = g / HC, hh = g % HC;
            const int l = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            float * xh = x + ((size_t) t * HC + hh) * D;
            const float * key = kv + (size_t) t * (HC + 1) * D + (size_t) hh * D;
            const float * w = wqk + (size_t) hh * D;
            float sx = 0.0f, sk = 0.0f, sd = 0.0f;
            for (int i = l; i < D; i += WG) {
                const float a = xh[i], b = key[i];
                sx += a * a;
                sk += b * b;
                sd += a * w[i] * b;
            }
            sx = sycl::reduce_over_group(sg, sx, sycl::plus<float>());
            sk = sycl::reduce_over_group(sg, sk, sycl::plus<float>());
            sd = sycl::reduce_over_group(sg, sd, sycl::plus<float>());
            if (lane == 0) { red[sgi] = sx; red[WG / 16 + sgi] = sk; red[2 * WG / 16 + sgi] = sd; }
            sycl::group_barrier(it.get_group());
            float tx = 0.0f, tk = 0.0f, td = 0.0f;
            for (int i = 0; i < WG / 16; ++i) { tx += red[i]; tk += red[WG / 16 + i]; td += red[2 * WG / 16 + i]; }
            const float rx = sycl::rsqrt(tx / (float) D + eps), rk = sycl::rsqrt(tk / (float) D + eps);
            const float dot = td * rx * rk / sycl::sqrt((float) D);
            const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
            const float sgn = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
            const float gate = 1.0f / (1.0f + sycl::exp(-(sgn * mag)));
            const float * val = kv + (size_t) t * (HC + 1) * D + (size_t) HC * D;
            sycl::group_barrier(it.get_group());   // every lane read xh above before any writes it
            for (int i = l; i < D; i += WG) xh[i] += val[i] * gate;
        });
    });
}

// ================================================================== routed experts: per card its jobs of the pass
// (one job per selected expert this card holds, its tokens merged), straight from the replicated routing - no host
struct TpExp {   // where (layer, expert) lives: the rank holding it (-1: the CPU pool) and, on that rank, its blob
    const uint8_t * blob = nullptr;
    uint32_t up_off = 0, down_off = 0;
    uint16_t gu_type = 0, d_type = 0, gu_row = 0, d_row = 0;
    int32_t owner = -1;
    int32_t pad = 0;
};

// jobs[0..nj_max): this card's experts in pick order (token-major), then null jobs (nt = 0: the expert kernels skip
// them - the pass graph launches them for nj_max); need_cpu[0] = whether a pick is on no card, need_cpu[1] = the
// real job count.  One work-item (passes of more than 64 picks; route_jobs below runs a lane a pick otherwise).
inline sycl::event route_jobs_serial(sycl::queue & q, const int32_t * ids, const float * wts, int n, int nu, const TpExp * tab, int me,
                                     strata::v41::xe::XeJob * jobs, int nj_max, int32_t * need_cpu) {
    return q.single_task([=] {
        int nj = 0;
        int32_t ex[64];
        bool cpu = false;
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < nu; ++i) {
                const int e = ids[t * nu + i];
                const TpExp & x = tab[e];
                if (x.owner != me) { if (x.owner < 0) cpu = true; continue; }
                int j = 0;
                while (j < nj && ex[j] != e) ++j;
                if (j == nj) {
                    if (nj >= nj_max || nj >= 64) continue;
                    ex[nj++] = e;
                    strata::v41::xe::XeJob & jb = jobs[j];
                    jb.blob = x.blob; jb.up_off = x.up_off; jb.down_off = x.down_off;
                    jb.gu_type = x.gu_type; jb.d_type = x.d_type; jb.gu_row = x.gu_row; jb.d_row = x.d_row;
                    jb.nt = 0;
                }
                strata::v41::xe::XeJob & jb = jobs[j];
                jb.tok[jb.nt] = t;
                jb.w[jb.nt] = wts[t * nu + i];
                jb.nt = (uint8_t) (jb.nt + 1);
            }
        for (int j = nj; j < nj_max; ++j) jobs[j].nt = 0;
        if (need_cpu) { need_cpu[0] = cpu ? 1 : 0; need_cpu[1] = nj; }
    });
}

// the same jobs, a lane a pick (the serial form's dependent loads were 4.5 us at 1 token, 15 us at 5): an expert's
// first pick opens its job (the job index = the first picks before it), and gathers the expert's later picks
inline sycl::event route_jobs(sycl::queue & q, const int32_t * ids, const float * wts, int n, int nu, const TpExp * tab, int me,
                              strata::v41::xe::XeJob * jobs, int nj_max, int32_t * need_cpu) {
    constexpr int WG = 64;
    const int np = n * nu;
    if (np > WG) return route_jobs_serial(q, ids, wts, n, nu, tab, me, jobs, nj_max, need_cpu);
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int32_t, 1> le(sycl::range<1>(WG), h);   // pick p's expert when this card holds it, else -1
        sycl::local_accessor<float, 1> lw(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const int p = (int) it.get_local_id(0);
            int e = -1;
            bool cpu = false;
            TpExp x;
            if (p < np) {
                const int ee = ids[p];
                x = tab[ee];
                if (x.owner == me) e = ee;
                else if (x.owner < 0) cpu = true;
                lw[p] = wts[p];
            }
            le[p] = e;
            sycl::group_barrier(it.get_group());
            bool first = e >= 0;
            for (int k = 0; k < p && first; ++k) first = le[k] != e;
            const int jidx = sycl::exclusive_scan_over_group(it.get_group(), first ? 1 : 0, sycl::plus<int>());
            const int nj = sycl::reduce_over_group(it.get_group(), first ? 1 : 0, sycl::plus<int>());
            const bool any_cpu = sycl::any_of_group(it.get_group(), cpu);
            if (first) {
                strata::v41::xe::XeJob & jb = jobs[jidx];
                jb.blob = x.blob; jb.up_off = x.up_off; jb.down_off = x.down_off;
                jb.gu_type = x.gu_type; jb.d_type = x.d_type; jb.gu_row = x.gu_row; jb.d_row = x.d_row;
                int nt = 0;
                for (int k = p; k < np; ++k)
                    if (le[k] == e) { jb.tok[nt] = k / nu; jb.w[nt] = lw[k]; ++nt; }
                jb.nt = (uint8_t) nt;
            }
            for (int j = nj + p; j < nj_max; j += WG) jobs[j].nt = 0;
            if (p == 0 && need_cpu) { need_cpu[0] = any_cpu ? 1 : 0; need_cpu[1] = nj; }
        });
    });
}

}  // namespace tpk

// ================================================================== the compressor of a kv source layer and the
// fake-quant roundings it uses (exact IEEE arithmetic: the pragma is file-scope only)
#pragma float_control(precise, on, push)
namespace tpk {

inline float fq_pow2_floor(float a) {
    const uint32_t u = sycl::bit_cast<uint32_t>(a);
    const uint32_t e = (u >> 23) & 0xffu;
    if (e == 0) return 0.0f;
    if (e == 0xffu) return a;
    return sycl::bit_cast<float>(e << 23);
}
inline float fq_round_fmt(float v, int mbits, float min_normal) {
    const float p = sycl::fmax(fq_pow2_floor(sycl::fabs(v)), min_normal);
    const float k = (float) (1 << mbits);
    const float r = sycl::rint(v / p * k) + 0.0f;
    return r / k * p;
}

// per token one 256-lane work-group (DH = 512: 2 values a lane, adjacent pairs):
//   ratio 2: latent_d = softmax-pool over the token's group of 2 states (rows of [state ring (2) | this pass's]),
//            kv / score planes from the [kv | gate] product y [n][2 DH]
//   ratio 1: latent = y [n][DH]
//   latent = rms_norm(latent) * cnorm; the indexer key k = rms_norm(attn_k . latent) * knorm, RoPE'd at cpos,
//   FP4_POW2_32 -> idxk[cslot] (bf16); the compressed row = RoPE(latent, cpos), FP4_E4M3_16 -> comp[cslot] (bf16)
struct CompArgs {
    const float * y = nullptr;            // ratio 2: [n][2 DH] (kv | gate); ratio 1: [n][DH]
    int ratio = 1;
    const float * st_kv = nullptr, * st_sc = nullptr;   // ratio 2: [2][DH] the state ring
    const int32_t * sidx = nullptr;       // ratio 2: [2 n]
    const float * cnorm = nullptr;        // [DH]
    const sycl::half * attn_k = nullptr;  // [ID][DH]
    const float * knorm = nullptr;        // [ID]
    const int32_t * cpos = nullptr, * cslot = nullptr;   // [n]
    tpdec::Rope rope;
    float eps = 0.0f;
    bool fq = true;
    uint16_t * comp = nullptr;            // bf16 rows [*][DH]
    uint16_t * idxk = nullptr;            // bf16 rows [*][ID]
    int n = 0;
};
constexpr int CDH = 512, CID = 128;

inline uint16_t f2bf(float f) {   // the values are FP4 / FP8 grid points (<= 7 significant bits): exact in bf16
    return (uint16_t) (sycl::bit_cast<uint32_t>(f) >> 16);
}

inline sycl::event comp_fin(sycl::queue & q, const CompArgs & a) {
    constexpr int WG = 256;
    const CompArgs A = a;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lat(sycl::range<1>(CDH), h);
        sycl::local_accessor<float, 1> kk(sycl::range<1>(CID), h);
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG / 16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) A.n * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int t = (int) it.get_group(0), l = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            float v[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int d = l + j * WG;
                if (A.ratio == 1) {
                    v[j] = A.y[(size_t) t * CDH + d];
                } else {
                    float vk[2], vs[2];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int row = A.sidx[2 * t + r];
                        vk[r] = row < 2 ? A.st_kv[row * CDH + d] : A.y[(size_t) (row - 2) * 2 * CDH + d];
                        vs[r] = row < 2 ? A.st_sc[row * CDH + d] : A.y[(size_t) (row - 2) * 2 * CDH + CDH + d];
                    }
                    const float m = sycl::fmax(vs[0], vs[1]);
                    const float e0 = sycl::exp(vs[0] - m), e1 = sycl::exp(vs[1] - m);
                    const float inv = 1.0f / (e0 + e1);
                    v[j] = vk[0] * (e0 * inv) + vk[1] * (e1 * inv);
                }
            }
            // rms_norm over DH (kv_fin's reduction at this shape)
            float tmp = v[0] * v[0] + v[1] * v[1];
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            if (lane == 0) red[sgi] = tmp;
            sycl::group_barrier(it.get_group());
            tmp = red[lane];
#pragma unroll
            for (int m = 8; m > 0; m >>= 1) tmp += sycl::permute_group_by_xor(sg, tmp, m);
            const float sc = sycl::rsqrt(tmp / CDH + A.eps);
            lat[l] = sc * v[0] * A.cnorm[l];
            lat[l + WG] = sc * v[1] * A.cnorm[l + WG];
            sycl::group_barrier(it.get_group());
            // the indexer key: 128 rows of attn_k . latent, 2 lanes a row (one sub-group: 8 rows)
            {
                const int r = l / 2, half = l % 2;
                const sycl::half * wr = A.attn_k + (size_t) r * CDH + half * (CDH / 2);
                float acc = 0.0f;
                for (int d = 0; d < CDH / 2; ++d) acc += (float) wr[d] * lat[half * (CDH / 2) + d];
                acc += sycl::permute_group_by_xor(sg, acc, 1);
                if (half == 0) kk[r] = acc;
            }
            sycl::group_barrier(it.get_group());
            // the compressed row: lane l holds the pair (2 l, 2 l + 1)
            const int cs = A.cslot[t], cp = A.cpos[t];
            {
                const int i0 = 2 * l;
                float x0 = lat[i0], x1 = lat[i0 + 1];
                const int off = CDH - A.rope.n_dims;
                if (i0 >= off) tpdec::rope_pair(A.rope, cp, i0 - off, true, x0, x1);
                if (A.fq) {   // FP4_E4M3_16: a 16-value block = 8 lanes (xor 1, 2, 4)
                    float am = sycl::fmax(sycl::fabs(x0), sycl::fabs(x1));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 1));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 2));
                    am = sycl::fmax(am, sycl::permute_group_by_xor(sg, am, 4));
                    const float s = fq_round_fmt(sycl::fmax(am, 6.0f / 512.0f) / 6.0f, 3, 0x1p-6f);
                    x0 = fq_round_fmt(sycl::fmin(sycl::fmax(x0 / s, -6.0f), 6.0f), 1, 1.0f) * s;
                    x1 = fq_round_fmt(sycl::fmin(sycl::fmax(x1 / s, -6.0f), 6.0f), 1, 1.0f) * s;
                }
                uint16_t * o = A.comp + (size_t) cs * CDH + i0;
                o[0] = f2bf(x0);
                o[1] = f2bf(x1);
            }
            // the indexer key: rms_norm * knorm over 128 (sub-groups 0-3: lane pairs), RoPE, FP4_POW2_32
            if (sgi < 4) {
                const int i0 = 2 * (sgi * 16 + lane);
                float x0 = kk[i0], x1 = kk[i0 + 1];
                float s2 = x0 * x0 + x1 * x1;
#pragma unroll
                for (int m = 8; m > 0; m >>= 1) s2 += sycl::permute_group_by_xor(sg, s2, m);
                red[sgi] = s2;   // (lanes write the same value)
            }
            sycl::group_barrier(it.get_group());
            if (sgi < 4) {
                const int i0 = 2 * (sgi * 16 + lane);
                const float ks = sycl::rsqrt((red[0] + red[1] + red[2] + red[3]) / CID + A.eps);
                float x0 = kk[i0] * ks * A.knorm[i0], x1 = kk[i0 + 1] * ks * A.knorm[i0 + 1];
                const int off = CID - A.rope.n_dims;
                if (i0 >= off) tpdec::rope_pair(A.rope, cp, i0 - off, true, x0, x1);
                if (A.fq) {   // FP4_POW2_32: a 32-value block = one sub-group
                    const float am = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(x0), sycl::fabs(x1)), sycl::maximum<float>());
                    const float s = tpdec::fq4_scale(am);
                    x0 = tpdec::fq4_apply(x0, s);
                    x1 = tpdec::fq4_apply(x1, s);
                }
                uint16_t * o = A.idxk + (size_t) cs * CID + i0;
                o[0] = f2bf(x0);
                o[1] = f2bf(x1);
            }
        });
    });
}

// the ratio-2 state write-back after every read of it: the pass's last min(n, 2) rows of the kv / score planes
inline sycl::event comp_state(sycl::queue & q, const float * y, int n, const int32_t * sslot, float * st_kv, float * st_sc) {
    const int k = n < 2 ? n : 2;
    return q.parallel_for(sycl::range<1>((size_t) k * CDH), [=](sycl::id<1> id) {
        const int j = (int) (id[0] / CDH), d = (int) (id[0] % CDH);
        const int t = n - k + j;
        st_kv[sslot[j] * CDH + d] = y[(size_t) t * 2 * CDH + d];
        st_sc[sslot[j] * CDH + d] = y[(size_t) t * 2 * CDH + CDH + d];
    });
}

}  // namespace tpk
#pragma float_control(pop)

// ================================================================== top-k by radix select (topk-radix.cpp, ggml-sycl's
// GGML_OP_TOP_K for large k): the winning column ids of each row, any order (sel_fin sorts them)
namespace tpk {

constexpr int TK_BITS = 8, TK_BUCKETS = 1 << TK_BITS, TK_COPIES = 8, TK_HIST = TK_BUCKETS * TK_COPIES;
constexpr int TK_SLM = TK_HIST + 5;
constexpr int TK_ROW_DONE = TK_BUCKETS, TK_ROW_PREFIX = TK_BUCKETS + 1, TK_ROW_MASK = TK_BUCKETS + 2,
              TK_ROW_NEED = TK_BUCKETS + 3, TK_ROW_GT = TK_BUCKETS + 4, TK_ROW_EQ = TK_BUCKETS + 5, TK_ROW_WORDS = TK_BUCKETS + 6;

inline uint32_t tk_key(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one work-group a row (rows of ncols < 64K)
inline sycl::event topk_rows(sycl::queue & q, const float * src, int64_t ld_src, int ncols, int nrows, int k, int32_t * dst,
                             int64_t ld_dst, int wg) {
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(TK_SLM), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * wg), sycl::range<1>((size_t) wg)), [=](sycl::nd_item<1> it) {
            using la = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
            const int row = (int) it.get_group(0);
            const float * s = src + (int64_t) row * ld_src;
            int32_t * di = dst + (int64_t) row * ld_dst;
            const int tid = (int) it.get_local_id(0), bsz = (int) it.get_local_range(0);
            uint32_t * hist = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t * s_digit = hist + TK_HIST, * s_bucket = hist + TK_HIST + 1, * s_need = hist + TK_HIST + 2;
            uint32_t * s_gt = hist + TK_HIST + 3, * s_eq = hist + TK_HIST + 4;
            if (tid == 0) { *s_gt = 0; *s_eq = 0; }
            const int copy = tid & (TK_COPIES - 1);
            uint32_t prefix = 0, mask = 0, need = (uint32_t) k;
            for (int shift = 32 - TK_BITS; shift >= 0; shift -= TK_BITS) {
                for (int i = tid; i < TK_HIST; i += bsz) hist[i] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                for (int col = tid; col < ncols; col += bsz) {
                    const uint32_t key = tk_key(s[col]);
                    if ((key & mask) == prefix) la(hist[((key >> shift) & (TK_BUCKETS - 1)) * TK_COPIES + copy]).fetch_add(1u);
                }
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t cnt = 0;
                if (tid < TK_BUCKETS) {
                    const uint32_t * hh = hist + (TK_BUCKETS - 1 - tid) * TK_COPIES;
                    for (int c = 0; c < TK_COPIES; c++) cnt += hh[c];
                }
                const uint32_t incl = sycl::inclusive_scan_over_group(it.get_group(), cnt, sycl::plus<uint32_t>());
                if (tid < TK_BUCKETS && incl >= need && incl - cnt < need) {
                    *s_digit = (uint32_t) (TK_BUCKETS - 1 - tid);
                    *s_bucket = cnt;
                    *s_need = need - (incl - cnt);
                }
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t digit = *s_digit, bucket_cnt = *s_bucket;
                need = *s_need;
                prefix |= digit << shift;
                mask |= (uint32_t) (TK_BUCKETS - 1) << shift;
                if (bucket_cnt == need) break;
                it.barrier(sycl::access::fence_space::local_space);
            }
            it.barrier(sycl::access::fence_space::local_space);
            const uint32_t base_eq = (uint32_t) k - need;
            for (int col = tid; col < ncols; col += bsz) {
                const uint32_t kp = tk_key(s[col]) & mask;
                if (kp > prefix) {
                    di[la(*s_gt).fetch_add(1u)] = col;
                } else if (kp == prefix) {
                    const uint32_t pos = la(*s_eq).fetch_add(1u);
                    if (pos < need) di[base_eq + pos] = col;
                }
            }
        });
    });
}

// rows split over `nparts` work-groups (ncols >= 64K): four digit passes + the emit, the state in global memory
// (state: nrows * TK_ROW_WORDS words, zero on entry; every launch leaves it zero again)
inline void topk_split(sycl::queue & q, const float * src, int64_t ld_src, int ncols, int nrows, int k, int32_t * dst,
                       int64_t ld_dst, int nparts, int wg, uint32_t * state) {
    using ga = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
    for (int shift = 32 - TK_BITS; shift >= 0; shift -= TK_BITS) {
        const bool first = shift == 32 - TK_BITS;
        q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(TK_HIST + 4), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * nparts * wg), sycl::range<1>((size_t) wg)), [=](sycl::nd_item<1> it) {
                using la = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
                const int g = (int) it.get_group(0), row = g / nparts, part = g % nparts;
                const float * s = src + (int64_t) row * ld_src;
                uint32_t * st = state + (int64_t) row * TK_ROW_WORDS;
                const int tid = (int) it.get_local_id(0), bsz = (int) it.get_local_range(0);
                uint32_t * hist = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * s_last = hist + TK_HIST, * s_row = hist + TK_HIST + 1;
                if (tid == 0) {
                    s_row[0] = first ? 0u : st[TK_ROW_PREFIX];
                    s_row[1] = first ? 0u : st[TK_ROW_MASK];
                    s_row[2] = first ? (uint32_t) k : st[TK_ROW_NEED];
                }
                for (int i = tid; i < TK_HIST; i += bsz) hist[i] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t prefix = s_row[0], mask = s_row[1], need = s_row[2];
                const int copy = tid & (TK_COPIES - 1);
                const int chunk = (ncols + nparts - 1) / nparts, col0 = part * chunk, col1 = sycl::min(ncols, col0 + chunk);
                for (int col = col0 + tid; col < col1; col += bsz) {
                    const uint32_t key = tk_key(s[col]);
                    if ((key & mask) == prefix) la(hist[((key >> shift) & (TK_BUCKETS - 1)) * TK_COPIES + copy]).fetch_add(1u);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int b = tid; b < TK_BUCKETS; b += bsz) {
                    uint32_t sum = 0;
                    for (int c = 0; c < TK_COPIES; c++) sum += hist[b * TK_COPIES + c];
                    if (sum) ga(st[b]).fetch_add(sum);
                }
                it.barrier(sycl::access::fence_space::global_and_local);
                if (tid == 0) {
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                    sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space> done(st[TK_ROW_DONE]);
                    *s_last = done.fetch_add(1u) == (uint32_t) (nparts - 1) ? 1u : 0u;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (*s_last == 0u) return;
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                uint32_t cnt = 0;
                if (tid < TK_BUCKETS) cnt = ga(st[TK_BUCKETS - 1 - tid]).load();
                const uint32_t incl = sycl::inclusive_scan_over_group(it.get_group(), cnt, sycl::plus<uint32_t>());
                if (tid < TK_BUCKETS && incl >= need && incl - cnt < need) {
                    const uint32_t digit = (uint32_t) (TK_BUCKETS - 1 - tid);
                    ga(st[TK_ROW_PREFIX]).store(prefix | (digit << shift));
                    ga(st[TK_ROW_MASK]).store(mask | ((uint32_t) (TK_BUCKETS - 1) << shift));
                    ga(st[TK_ROW_NEED]).store(need - (incl - cnt));
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int b = tid; b < TK_BUCKETS; b += bsz) ga(st[b]).store(0u);
                if (tid == 0) ga(st[TK_ROW_DONE]).store(0u);
            });
        });
    }
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * nparts * wg), sycl::range<1>((size_t) wg)), [=](sycl::nd_item<1> it) {
            using la = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
            const int g = (int) it.get_group(0), row = g / nparts, part = g % nparts;
            const float * s = src + (int64_t) row * ld_src;
            int32_t * di = dst + (int64_t) row * ld_dst;
            uint32_t * st = state + (int64_t) row * TK_ROW_WORDS;
            const int tid = (int) it.get_local_id(0), bsz = (int) it.get_local_range(0);
            uint32_t * sl = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t * s_gt = sl, * s_eq = sl + 1, * s_bgt = sl + 2, * s_beq = sl + 3, * s_row = sl + 4;
            if (tid == 0) {
                *s_gt = 0; *s_eq = 0;
                s_row[0] = st[TK_ROW_PREFIX]; s_row[1] = st[TK_ROW_MASK]; s_row[2] = st[TK_ROW_NEED];
            }
            it.barrier(sycl::access::fence_space::local_space);
            const uint32_t prefix = s_row[0], mask = s_row[1], need = s_row[2];
            const uint32_t base_eq = (uint32_t) k - need;
            const int chunk = (ncols + nparts - 1) / nparts, col0 = part * chunk, col1 = sycl::min(ncols, col0 + chunk);
            for (int col = col0 + tid; col < col1; col += bsz) {
                const uint32_t kp = tk_key(s[col]) & mask;
                if (kp > prefix) la(*s_gt).fetch_add(1u);
                else if (kp == prefix) la(*s_eq).fetch_add(1u);
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tid == 0) {
                const uint32_t n_gt = *s_gt, n_eq = *s_eq;
                *s_bgt = n_gt ? ga(st[TK_ROW_GT]).fetch_add(n_gt) : 0u;
                *s_beq = n_eq ? ga(st[TK_ROW_EQ]).fetch_add(n_eq) : 0u;
                *s_gt = 0; *s_eq = 0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const uint32_t bgt = *s_bgt, beq = *s_beq;
            for (int col = col0 + tid; col < col1; col += bsz) {
                const uint32_t kp = tk_key(s[col]) & mask;
                if (kp > prefix) {
                    di[bgt + la(*s_gt).fetch_add(1u)] = col;
                } else if (kp == prefix) {
                    const uint32_t pos = beq + la(*s_eq).fetch_add(1u);
                    if (pos < need) di[base_eq + pos] = col;
                }
            }
        });
    });
    // the emit counters back to zero for the next call (the digit passes cleared the rest)
    q.parallel_for(sycl::range<1>((size_t) nrows), [=](sycl::id<1> r) {
        state[(int64_t) r[0] * TK_ROW_WORDS + TK_ROW_GT] = 0u;
        state[(int64_t) r[0] * TK_ROW_WORDS + TK_ROW_EQ] = 0u;
    });
}

}  // namespace tpk
