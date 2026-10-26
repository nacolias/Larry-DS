#!/usr/bin/env python3
# patch-sdpao16.py [XOLOTL_DIR] - larry, STRATA_SDPA_O16 (engine, default 0; with STRATA_OPROJ_A): the prompt attention's
# output in fp16.  GGML_OP_DSV41_ATTN_SDPA widened its fp16 result O' to f32 o (67 MB written a 512-column span), the
# de-RoPE ran on that, and ggml_dsv41_oproj_a converted o back to fp16 for wo_a (67 MB read, 34 MB written).
# ggml_dsv41_attn_sdpa_o16 (ggml.h / ggml.c: the same op, an F16 result): the oneDNN path copies O' through, the oneMKL
# path's second gemm_batch writes fp16; the in-place de-RoPE (STRATA_ROPE_INPLACE) takes F16; ggml_dsv41_oproj_a takes
# an F16 o as wo_a's input as it is.  The same values: wo_a always read o rounded to fp16.  Needs patch-oproja.py.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
G = os.path.join(X, "third_party/llama.cpp/ggml")


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


edit(os.path.join(G, "include/ggml.h"), [(
"""    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa(""",
"""    // larry: ggml_dsv41_attn_sdpa with an F16 result (patch-sdpao16.py)
    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa_o16(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * kp,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * sinks,
            float                 scale);
    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa(""")], "ggml_dsv41_attn_sdpa_o16")

edit(os.path.join(G, "src/ggml.c"), [
("""struct ggml_tensor * ggml_dsv41_kgather2(""",
"""// larry, patch-sdpao16.py: ggml_dsv41_attn_sdpa with an F16 result (its consumers: the in-place de-RoPE, ggml_dsv41_oproj_a)
struct ggml_tensor * ggml_dsv41_attn_sdpa_o16(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * kp,
        struct ggml_tensor  * mask,
        struct ggml_tensor  * sinks,
        float                 scale) {
    struct ggml_tensor * r = ggml_dsv41_attn_sdpa(ctx, q, kp, mask, sinks, scale);
    r->type = GGML_TYPE_F16;
    r->nb[0] = ggml_type_size(GGML_TYPE_F16);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) r->nb[i] = r->nb[i - 1] * r->ne[i - 1];
    return r;
}

struct ggml_tensor * ggml_dsv41_kgather2("""),
("""struct ggml_tensor * ggml_dsv41_oproj_a(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a) {
    GGML_ASSERT(o->type == GGML_TYPE_F32 && ggml_is_contiguous(o));""",
"""struct ggml_tensor * ggml_dsv41_oproj_a(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a) {
    GGML_ASSERT((o->type == GGML_TYPE_F32 || o->type == GGML_TYPE_F16) && ggml_is_contiguous(o));   // (F16: patch-sdpao16.py)"""),
], "ggml_dsv41_attn_sdpa_o16")

edit(os.path.join(G, "src/ggml-sycl/fattn-dsv4.cpp"), [
("""            // O' f16 [n][NH][Dp] -> dst f32 [n][NH][D]
            const sycl::half * op = Op.get();
            float * o = (float *) dst->data;
            stream->parallel_for(sycl::range<1>((size_t) n * NH * (D / 4)), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int64_t v = (int64_t) (i % (size_t) (D / 4)), bh = (int64_t) (i / (size_t) (D / 4));
                const sycl::half4 h = *(const sycl::half4 *) (op + bh * Dp + v * 4);
                *(sycl::float4 *) (o + bh * D + v * 4) = sycl::float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
            });""",
"""            // O' f16 [n][NH][Dp] -> dst f32 [n][NH][D] (an F16 dst, ggml_dsv41_attn_sdpa_o16: O' through as it is)
            const sycl::half * op = Op.get();
            if (dst->type == GGML_TYPE_F16) {
                sycl::half * o16 = (sycl::half *) dst->data;
                stream->parallel_for(sycl::range<1>((size_t) n * NH * (D / 4)), [=](sycl::id<1> id) {
                    const size_t i = id[0];
                    const int64_t v = (int64_t) (i % (size_t) (D / 4)), bh = (int64_t) (i / (size_t) (D / 4));
                    *(sycl::half4 *) (o16 + bh * D + v * 4) = *(const sycl::half4 *) (op + bh * Dp + v * 4);
                });
            } else {
            float * o = (float *) dst->data;
            stream->parallel_for(sycl::range<1>((size_t) n * NH * (D / 4)), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int64_t v = (int64_t) (i % (size_t) (D / 4)), bh = (int64_t) (i / (size_t) (D / 4));
                const sycl::half4 h = *(const sycl::half4 *) (op + bh * Dp + v * 4);
                *(sycl::float4 *) (o + bh * D + v * 4) = sycl::float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
            });
            }"""),
("""    mkl::blas::row_major::gemm_batch(*stream, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, 1.0f, p, M,
                                     NH * M, kp, Dp, Mp * Dp, 0.0f, (float *) dst->data, D, NH * D, n);
}""",
"""    if (dst->type == GGML_TYPE_F16)   // larry, patch-sdpao16.py
        mkl::blas::row_major::gemm_batch(*stream, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, sycl::half(1.0f), p, M,
                                         NH * M, kp, Dp, Mp * Dp, sycl::half(0.0f), (sycl::half *) dst->data, D, NH * D, n);
    else
    mkl::blas::row_major::gemm_batch(*stream, mkl::transpose::nontrans, mkl::transpose::nontrans, NH, D, M, 1.0f, p, M,
                                     NH * M, kp, Dp, Mp * Dp, 0.0f, (float *) dst->data, D, NH * D, n);
}"""),
], "patch-sdpao16.py")

edit(os.path.join(G, "src/ggml-sycl/ggml-sycl.cpp"), [
("""static void larry_slices_batched(ggml_backend_sycl_context & ctx, const char * w0, size_t sb, int64_t N, int64_t K, int64_t G,
                                 const float * xs, int64_t ncols, float * dd, int64_t ldc, int64_t sc) {
    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> xa(ctx.pool(), (size_t) (ncols * G * K));""",
"""static void larry_slices_batched(ggml_backend_sycl_context & ctx, const char * w0, size_t sb, int64_t N, int64_t K, int64_t G,
                                 const float * xs, int64_t ncols, float * dd, int64_t ldc, int64_t sc,
                                 const sycl::half * x16 = nullptr) {   // x16: the input already fp16 (patch-sdpao16.py)
    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> xa(ctx.pool(), x16 ? (size_t) 1 : (size_t) (ncols * G * K));"""),
("""    sycl::half * xh = xa.get();
    const size_t ne4 = (size_t) (ncols * G * K) / 4;
    st->parallel_for(sycl::range<1>(ne4), [=](sycl::id<1> i) {
        const sycl::float4 v = *(const sycl::float4 *) (xs + 4 * i[0]);
        *(sycl::vec<sycl::half, 4> *) (xh + 4 * i[0]) =
            sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
    });
    if (sc >= ldc * ncols) {""",
"""    const sycl::half * xh = x16 ? x16 : xa.get();
    if (!x16) {
        sycl::half * xw = xa.get();
        const size_t ne4 = (size_t) (ncols * G * K) / 4;
        st->parallel_for(sycl::range<1>(ne4), [=](sycl::id<1> i) {
            const sycl::float4 v = *(const sycl::float4 *) (xs + 4 * i[0]);
            *(sycl::vec<sycl::half, 4> *) (xw + 4 * i[0]) =
                sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        });
    }
    if (sc >= ldc * ncols) {"""),
("""    larry_slices_batched(ctx, (const char *) wa->data, wa->nb[2], N, K, G, (const float *) o->data, n, (float *) dst->data,
                         N * G, N);
    return true;""",
"""    const bool o16 = o->type == GGML_TYPE_F16;   // (patch-sdpao16.py: the attention's fp16 output as it is)
    larry_slices_batched(ctx, (const char *) wa->data, wa->nb[2], N, K, G, o16 ? nullptr : (const float *) o->data, n,
                         (float *) dst->data, N * G, N, o16 ? (const sycl::half *) o->data : nullptr);
    return true;"""),
], "patch-sdpao16.py")

edit(os.path.join(X, "tools/v41/engine/ds41_dense.cpp"), [(
"""        o = ggml_dsv41_attn_sdpa(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"), 1.0f / std::sqrt((float) DH));   // [DH, NH, n]""",
"""        // larry, STRATA_SDPA_O16 (patch-sdpao16.py, default 0): the output in fp16 when wo_a takes it as one op (STRATA_OPROJ_A)
        static const bool sdpa_o16 = env_on("STRATA_SDPA_O16", false) && env_on("STRATA_OPROJ_A", false);
        const bool o16 = sdpa_o16 && im.sycl && n > 8 && im.BL(il, "attn_output_a.weight")->type == GGML_TYPE_Q6_K;
        o = (o16 ? ggml_dsv41_attn_sdpa_o16 : ggml_dsv41_attn_sdpa)(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"),
                                                                   1.0f / std::sqrt((float) DH));   // [DH, NH, n]""")],
     "STRATA_SDPA_O16")
