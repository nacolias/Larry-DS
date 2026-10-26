#!/usr/bin/env python3
# patch-hcwide.py [XOLOTL_DIR] - larry: the fused hc ops at prompt widths.  STRATA_HC_FUSE_MAXN (engine, default 8)
# keeps the prompt path off ggml_dsv41_hc_fa / _hc_fb (hc_post + the mixes' dots + hc_pre in one pass, outputs written
# in place): at 512-token spans hc_fa lost to the composite (31.6K prompt 26.2 -> 27.7 s) - its sub-group per (token,
# 128-column chunk, row group of 6) re-read the token's state for each of 4 row groups and every token read all of
# fn [24 x 20480] through L2.  The composite now costs ~1.3 ms a 512-column span (abwT122: HC_POST 2 x 173, HC_PRE
# 2 x 120, the mixes 2 x 153 us, the state copies ~0.4 ms).  Here hc_fa's wide form (n > 8; GGML_SYCL_HC_FA_WIDE=0: the
# narrow one): a work-group = 16 tokens x one 128-column chunk, the chunk's fn rows staged in local memory once, a
# sub-group = one token with all 24 rows - the same outputs (h, z, the chunk partials) and arithmetic.  hc_merge (a
# kernel for <= 64 tokens) stays off past 64.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
F = os.path.join(X, "third_party/llama.cpp/ggml/src/ggml-sycl/dsv41-fused.cpp")
D = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")


def edit(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:70]))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


wide = r'''// larry, patch-hcwide.py: GGML_OP_DSV41_HC_FA at prompt widths - a work-group = 16 tokens x one 128-column chunk; the
// chunk's fn rows (24 mixes x 4 hc rows x 128 columns) staged in local memory once; a sub-group = one token, all 24
// rows (hc_fa_part's 4 row groups each re-read the state and redid hc_post / hc_pre); h, z and the partials as there
template <bool W16, bool POST>
void hc_fa_wide_sycl(sycl::queue * q, const float * x, int64_t xs1, int64_t xs2, const void * fn, int64_t fs1,
                     const float * pre, int64_t ps1, const float * y, int64_t ys1, const float * post, int64_t pos1,
                     const float * comb, int64_t cs1, int64_t cs2, float * out, int64_t D, int64_t n, float * hdst,
                     const float * y2, int64_t y2s1) {
    constexpr int HC = 4, NR = 24, IPL = 8, SG = 16, NV = 32, TT = 16, CW = SG * IPL;
    using WT = typename std::conditional<W16, sycl::half, float>::type;
    const int64_t nch = D / CW, ntb = (n + TT - 1) / TT;
    q->submit([&](sycl::handler & h) {
        sycl::local_accessor<WT, 1> ws(sycl::range<1>((size_t) NR * HC * CW), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (ntb * nch * TT * SG)), sycl::range<1>(TT * SG)),
                       sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), lid = (int) it.get_local_id(0);
            const int64_t g = (int64_t) it.get_group(0), ch = g % nch, t = (g / nch) * TT + (int64_t) sg.get_group_linear_id();
            for (int i = lid * IPL; i < NR * HC * CW; i += TT * SG * IPL) {   // ws[(r * HC + d) * CW + c]
                const int rd = i / CW, c = i % CW;
                const WT * src = (const WT *) fn + (int64_t) (rd / HC) * fs1 + (int64_t) (rd % HC) * D + ch * CW + c;
                *(sycl::vec<WT, IPL> *) &ws[i] = *(const sycl::vec<WT, IPL> *) src;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (t >= n) return;   // (after the barrier; uniform in a sub-group)
            float * oh = out;
            float * oz = out + n * HC * D;
            float * op = oz + n * D;
            const int64_t i0 = ch * CW + lane * IPL;
            float hv[HC][IPL];
            const float * xt = x + t * xs2 + i0;
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                const sycl::float4 va = *(const sycl::float4 *) (xt + s * xs1), vb = *(const sycl::float4 *) (xt + s * xs1 + 4);
                hv[s][0] = va.x(); hv[s][1] = va.y(); hv[s][2] = va.z(); hv[s][3] = va.w();
                hv[s][4] = vb.x(); hv[s][5] = vb.y(); hv[s][6] = vb.z(); hv[s][7] = vb.w();
            }
            if constexpr (POST) {
                float xr[HC][IPL];
#pragma unroll
                for (int s = 0; s < HC; ++s)
#pragma unroll
                    for (int j = 0; j < IPL; ++j) xr[s][j] = hv[s][j];
                sycl::float4 ya = *(const sycl::float4 *) (y + t * ys1 + i0), yb = *(const sycl::float4 *) (y + t * ys1 + i0 + 4);
                if (y2) {
                    ya += *(const sycl::float4 *) (y2 + t * y2s1 + i0);
                    yb += *(const sycl::float4 *) (y2 + t * y2s1 + i0 + 4);
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
                        hv[d][j] = v;
                    }
                }
            }
            const float * pw = pre + t * ps1;
            const float p0 = pw[0], p1 = pw[1], p2 = pw[2], p3 = pw[3];
            float z[IPL];
#pragma unroll
            for (int j = 0; j < IPL; ++j) {
                float v = hv[0][j] * p0;
                v += hv[1][j] * p1;
                v += hv[2][j] * p2;
                v += hv[3][j] * p3;
                z[j] = v;
            }
            float * P = op + (t * nch + ch) * NV;
            if constexpr (POST) {
#pragma unroll
                for (int d = 0; d < HC; ++d) {
                    float * o = oh + t * HC * D + d * D + i0;
                    *(sycl::float4 *) o = sycl::float4(hv[d][0], hv[d][1], hv[d][2], hv[d][3]);
                    *(sycl::float4 *) (o + 4) = sycl::float4(hv[d][4], hv[d][5], hv[d][6], hv[d][7]);
                    if (hdst) {
                        float * o2 = hdst + t * HC * D + d * D + i0;
                        *(sycl::float4 *) o2 = sycl::float4(hv[d][0], hv[d][1], hv[d][2], hv[d][3]);
                        *(sycl::float4 *) (o2 + 4) = sycl::float4(hv[d][4], hv[d][5], hv[d][6], hv[d][7]);
                    }
                }
            }
            *(sycl::float4 *) (oz + t * D + i0) = sycl::float4(z[0], z[1], z[2], z[3]);
            *(sycl::float4 *) (oz + t * D + i0 + 4) = sycl::float4(z[4], z[5], z[6], z[7]);
            float sh = 0.0f, sz = 0.0f;
#pragma unroll
            for (int j = 0; j < IPL; ++j) {
#pragma unroll
                for (int d = 0; d < HC; ++d) sh += hv[d][j] * hv[d][j];
                sz += z[j] * z[j];
            }
            sh = sycl::reduce_over_group(sg, sh, sycl::plus<float>());
            sz = sycl::reduce_over_group(sg, sz, sycl::plus<float>());
            if (lane == 0) { P[NR] = sh; P[NR + 1] = sz; }
            for (int r = 0; r < NR; ++r) {
                float a = 0.0f;
#pragma unroll
                for (int d = 0; d < HC; ++d) {
                    const sycl::vec<WT, IPL> wv = *(const sycl::vec<WT, IPL> *) &ws[(r * HC + d) * CW + lane * IPL];
#pragma unroll
                    for (int j = 0; j < IPL; ++j) a += (float) wv[j] * hv[d][j];
                }
                a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                if (lane == 0) P[r] = a;
            }
        });
    });
}

}  // namespace

void ggml_sycl_op_dsv41_hc_fa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {'''

edit(F, [
("""#include <cmath>
#include <cstdint>""", """#include <cmath>
#include <cstdint>
#include <type_traits>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>   // (patch-hcwide.py)"""),
("""}  // namespace

void ggml_sycl_op_dsv41_hc_fa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {""", wide),
("""    float * out = (float *) dst->data;
    sycl::queue * q = ctx.stream();
    if (y) {
        const float * py = (const float *) y->data;
        const float * po = (const float *) post->data;
        const float * pc = (const float *) comb->data;
        const int64_t ys1 = y->nb[1] / 4, pos1 = post->nb[1] / 4, cs1 = comb->nb[1] / 4, cs2 = comb->nb[2] / 4;
        float * hd = dst->src[6] ? (float *) dst->src[6]->data : nullptr;
        const float * py2 = dst->src[7] ? (const float *) dst->src[7]->data : nullptr;
        const int64_t y2s1 = dst->src[7] ? dst->src[7]->nb[1] / 4 : 0;
        if (w16) hc_fa_sycl<true, true>""",
"""    float * out = (float *) dst->data;
    sycl::queue * q = ctx.stream();
    static const bool wide = [] { const char * v = std::getenv("GGML_SYCL_HC_FA_WIDE"); return !(v && v[0] == '0'); }();
    if (wide && n > 8) {   // larry, patch-hcwide.py: prompt widths
        const float * py = y ? (const float *) y->data : nullptr;
        const float * po = y ? (const float *) post->data : nullptr;
        const float * pc = y ? (const float *) comb->data : nullptr;
        const int64_t ys1 = y ? y->nb[1] / 4 : 0, pos1 = y ? post->nb[1] / 4 : 0, cs1 = y ? comb->nb[1] / 4 : 0, cs2 = y ? comb->nb[2] / 4 : 0;
        float * hd = y && dst->src[6] ? (float *) dst->src[6]->data : nullptr;
        const float * py2 = y && dst->src[7] ? (const float *) dst->src[7]->data : nullptr;
        const int64_t y2s1 = y && dst->src[7] ? dst->src[7]->nb[1] / 4 : 0;
        if (y) {
            if (w16) hc_fa_wide_sycl<true, true>(q, px, xs1, xs2, fn->data, fs1, pp, ps1, py, ys1, po, pos1, pc, cs1, cs2, out, D, n, hd, py2, y2s1);
            else     hc_fa_wide_sycl<false, true>(q, px, xs1, xs2, fn->data, fs1, pp, ps1, py, ys1, po, pos1, pc, cs1, cs2, out, D, n, hd, py2, y2s1);
        } else {
            if (w16) hc_fa_wide_sycl<true, false>(q, px, xs1, xs2, fn->data, fs1, pp, ps1, nullptr, 0, nullptr, 0, nullptr, 0, 0, out, D, n, nullptr, nullptr, 0);
            else     hc_fa_wide_sycl<false, false>(q, px, xs1, xs2, fn->data, fs1, pp, ps1, nullptr, 0, nullptr, 0, nullptr, 0, 0, out, D, n, nullptr, nullptr, 0);
        }
        return;
    }
    if (y) {
        const float * py = (const float *) y->data;
        const float * po = (const float *) post->data;
        const float * pc = (const float *) comb->data;
        const int64_t ys1 = y->nb[1] / 4, pos1 = post->nb[1] / 4, cs1 = comb->nb[1] / 4, cs2 = comb->nb[2] / 4;
        float * hd = dst->src[6] ? (float *) dst->src[6]->data : nullptr;
        const float * py2 = dst->src[7] ? (const float *) dst->src[7]->data : nullptr;
        const int64_t y2s1 = dst->src[7] ? dst->src[7]->nb[1] / 4 : 0;
        if (w16) hc_fa_sycl<true, true>"""),
], "patch-hcwide.py")

edit(D, [(
"""    ggml_tensor * hc_merge(ggml_tensor * fb) const {
        static const bool on = env_on("STRATA_HC_MERGE", true);
        return on && g->n_embd % 512 == 0 ? ggml_dsv41_hc_merge(fb) : fb;""",
"""    ggml_tensor * hc_merge(ggml_tensor * fb) const {
        static const bool on = env_on("STRATA_HC_MERGE", true);
        // (larry, patch-hcwide.py: the merged kernel takes <= 64 tokens - hc_fb's op_params[2] is its n)
        return on && g->n_embd % 512 == 0 && fb->op_params[2] <= 64 ? ggml_dsv41_hc_merge(fb) : fb;""")], "patch-hcwide.py")
