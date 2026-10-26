#!/usr/bin/env python3
# patch-normmvw.py [XOLOTL_DIR] - larry, STRATA_HC_NORM_MV_WIDE (engine, default 0): the prompt path's hc mixes through
# ggml_dsv41_norm_mv, with a wide form of its ggml-sycl kernel.  At 512-token spans the mixes were rms_norm over the
# 4-stream state [20480 x 512] f32 (197 us: reads 42 MB, writes 42 MB) and a [24 x 20480] f32 GEMM on the normed copy
# (126 us), twice a layer-span - 6.8 % of card 0 in a 31K prompt's second batch (abwT118).  The op folds the norm out of
# the product (out = (w . x) * rsqrt(mean(x^2) + eps)); its decode kernel (a work-group per token and output row) reads
# each token's state once per row.  The wide one reads it once: a sub-group = one token, its lanes walk 4 values at a
# time keeping all R <= 24 dots and the sum of squares, 16 tokens a work-group share the weight through L1, the inner
# dimension split over work-groups (partials, then one pass for the sums and the scale).  f32 throughout.
# GGML_SYCL_NORM_MV_WIDE=0: the per-row kernel at every width.  Idempotent.
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


kernel = r"""// larry, patch-normmvw.py: GGML_OP_DSV41_NORM_MV at prompt widths - each token's x read ONCE for all its R <= 24
// outputs: a work-group = 64 tokens x a 640-value slice of the inner dimension (C / 640 slices); the slice's weight
// rows staged in local memory once (float4 loads, ~8 a work-item); a sub-group = 2 tokens (lanes 4 values apart, 2 x R
// dots and the sums of squares in registers: every weight read serves both); partials [KS][n][32], then one pass:
// out = sum(dots) * rsqrt(sum(squares) / C + eps).  (Forms that lost: no staging - the weight through L2 for every
// token, 349 us a call; 16 tokens a work-group with scalar staging - 120 dependent loads a work-item, 604 us.)
template <bool W16>
static void norm_mv_wide(ggml_backend_sycl_context & ctx, const void * pw, const float * px, float * pd, int64_t C, int64_t R,
                         int64_t n, float eps) {
    constexpr int TOK = 64, RM = 24, SGS = 32, CS = 640;
    queue_ptr stream = ctx.stream();
    const int64_t nwt = (n + TOK - 1) / TOK, KS = C / CS;
    ggml_sycl_pool_alloc<float> part(ctx.pool(), (size_t) (KS * n * 32));
    float * pp = part.get();
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> ws(sycl::range<1>((size_t) RM * CS), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nwt * KS * SGS * 16)), sycl::range<1>(SGS * 16)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), lid = (int) it.get_local_id(0);
            const int64_t g = (int64_t) it.get_group(0), ks = g % KS, c0 = ks * CS;
            for (int i = lid * 4; i < (int) R * CS; i += SGS * 16 * 4) {   // the slice's weight rows, once a work-group
                const int r = i / CS, c = i % CS;
                sycl::float4 v;
                if constexpr (W16) {
                    const sycl::vec<sycl::half, 4> hv = *(const sycl::vec<sycl::half, 4> *) ((const sycl::half *) pw + r * C + c0 + c);
                    v = sycl::float4((float) hv[0], (float) hv[1], (float) hv[2], (float) hv[3]);
                } else {
                    v = *(const sycl::float4 *) ((const float *) pw + r * C + c0 + c);
                }
                *(sycl::float4 *) &ws[i] = v;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int64_t t0 = (g / KS) * TOK + 2 * (int64_t) sg.get_group_linear_id(), t1 = t0 + 1;
            if (t0 >= n) return;   // (after the barrier: a sub-group's exit is uniform)
            const bool two = t1 < n;
            const float * x0 = px + t0 * C + c0;
            const float * x1 = px + (two ? t1 : t0) * C + c0;
            float a0[RM], a1[RM];
#pragma unroll
            for (int r = 0; r < RM; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
            float q0 = 0.0f, q1 = 0.0f;
            for (int c = lane * 4; c < CS; c += 64) {
                const sycl::float4 u = *(const sycl::float4 *) (x0 + c), v = *(const sycl::float4 *) (x1 + c);
                q0 += u.x() * u.x() + u.y() * u.y() + u.z() * u.z() + u.w() * u.w();
                q1 += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
#pragma unroll
                for (int r = 0; r < RM; ++r) {
                    if (r < R) {
                        const float * wr = &ws[r * CS + c];
                        const float w0 = wr[0], w1 = wr[1], w2 = wr[2], w3 = wr[3];
                        a0[r] += w0 * u.x() + w1 * u.y() + w2 * u.z() + w3 * u.w();
                        a1[r] += w0 * v.x() + w1 * v.y() + w2 * v.z() + w3 * v.w();
                    }
                }
            }
            float * p0 = pp + (ks * n + t0) * 32;
            float * p1 = pp + (ks * n + t1) * 32;
            q0 = sycl::reduce_over_group(sg, q0, sycl::plus<float>());
            q1 = sycl::reduce_over_group(sg, q1, sycl::plus<float>());
            if (lane == 0) { p0[RM] = q0; if (two) p1[RM] = q1; }
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                if (r < R) {
                    const float s0 = sycl::reduce_over_group(sg, a0[r], sycl::plus<float>());
                    const float s1 = sycl::reduce_over_group(sg, a1[r], sycl::plus<float>());
                    if (lane == 0) { p0[r] = s0; if (two) p1[r] = s1; }
                }
            }
        });
    });
    stream->parallel_for(sycl::range<1>((size_t) (n * R)), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0] / R, r = (int64_t) id[0] % R;
        float d = 0.0f, q = 0.0f;
        for (int64_t k = 0; k < KS; ++k) { d += pp[(k * n + t) * 32 + r]; q += pp[(k * n + t) * 32 + RM]; }
        pd[t * R + r] = d * sycl::rsqrt(q / (float) C + eps);
    });
}

// GGML_OP_DSV41_NORM_MV (larry): a work-group per (token, output row): its lanes walk the row 8 values at a time,"""

edit(F, [
("// GGML_OP_DSV41_NORM_MV (larry): a work-group per (token, output row): its lanes walk the row 8 values at a time,", kernel),
("""    const bool w16 = w->type == GGML_TYPE_F16;
    const void * pw = w->data;
    const float * px = (const float *) x->data;
    float * pd = (float *) dst->data;
    constexpr int WG = 256;""",
"""    const bool w16 = w->type == GGML_TYPE_F16;
    const void * pw = w->data;
    const float * px = (const float *) x->data;
    float * pd = (float *) dst->data;
    static const bool wide = [] { const char * v = std::getenv("GGML_SYCL_NORM_MV_WIDE"); return !(v && v[0] == '0'); }();
    if (wide && n > 8 && R <= 24 && C % 640 == 0) {   // larry, patch-normmvw.py: prompt widths
        if (w16) norm_mv_wide<true>(ctx, pw, px, pd, C, R, n, eps);
        else norm_mv_wide<false>(ctx, pw, px, pd, C, R, n, eps);
        return;
    }
    constexpr int WG = 256;"""),
], "patch-normmvw.py")

edit(D, [
("""        static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
        ggml_tensor * mixes = nullptr;
        if (dec_fuse && im->sycl && nt <= 8 && ggml_is_contiguous(x)) {""",
"""        static const bool dec_fuse = env_on("STRATA_DEC_FUSE", true);
        // larry, STRATA_HC_NORM_MV_WIDE (patch-normmvw.py, default 0): prompt widths too - ggml-sycl's wide norm_mv reads
        // the state once (the rms_norm copy + GEMM read it twice and wrote it once: 323 -> ~100 us at 512 tokens)
        static const bool nmv_wide = env_on("STRATA_HC_NORM_MV_WIDE", false);
        ggml_tensor * mixes = nullptr;
        if (dec_fuse && im->sycl && (nt <= 8 || nmv_wide) && ggml_is_contiguous(x)) {"""),
], "STRATA_HC_NORM_MV_WIDE")
