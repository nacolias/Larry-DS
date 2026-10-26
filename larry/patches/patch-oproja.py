#!/usr/bin/env python3
# patch-oproja.py [XOLOTL_DIR] - larry, STRATA_OPROJ_A (engine, default 0): the prompt path's wo_a as ONE op writing wo_b's
# input layout.  ggml_mul_mat(wo_a, permute(o)) gave [OL, n, OG], permuted back and copied for wo_b (CONT [8192 x 512]:
# 162 us a 512-column span, abwT118) on top of wo_a's own product.  ggml_dsv41_oproj_a (ggml.h / ggml.c: a
# GGML_OP_DSV41_OPROJ node with op_params[0] = 1 and no wo_b) -> F32 [OL * OG, n]; ggml-sycl runs it as
# patch-slicesbatch.py's gemm_batch (o converted to fp16 once, the per-slice planar slices dequantised into one fp16
# cache entry) with wo_b's layout as the output strides.  Needs patch-slicesbatch.py first.  Idempotent.
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
"""    // larry: top-k ids (I32 [K, n], K <= 512) sorted ascending into sel (I32 [K, n]) and selm (F32 [K, n]: 0 where""",
"""    // larry: wo_a alone at prompt widths, written in wo_b's input layout - o F32 (contiguous: a column's OG groups of OGD
    // values), wo_a [OGD, OL, OG] (a 3-D view of a per-slice planar Q6_K weight) -> F32 [OL * OG, n] (column t: group g's
    // OL outputs at g * OL).  ggml-sycl only (GGML_OP_DSV41_OPROJ, op_params[0] = 1).
    GGML_API struct ggml_tensor * ggml_dsv41_oproj_a(
            struct ggml_context * ctx,
            struct ggml_tensor  * o,
            struct ggml_tensor  * wo_a);
    // larry: top-k ids (I32 [K, n], K <= 512) sorted ascending into sel (I32 [K, n]) and selm (F32 [K, n]: 0 where""")],
     "ggml_dsv41_oproj_a")

edit(os.path.join(G, "src/ggml.c"), [(
"""static void ggml_dsv41_set_rope(struct ggml_tensor * t, int i0, struct ggml_dsv41_rope r) {""",
"""// larry: wo_a alone, in wo_b's input layout (GGML_OP_DSV41_OPROJ with op_params[0] = 1, no wo_b)
struct ggml_tensor * ggml_dsv41_oproj_a(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a) {
    GGML_ASSERT(o->type == GGML_TYPE_F32 && ggml_is_contiguous(o));
    const int64_t OGD = wo_a->ne[0], OL = wo_a->ne[1], OG = wo_a->ne[2];
    GGML_ASSERT(ggml_nelements(o) % (OGD * OG) == 0);
    const int64_t n = ggml_nelements(o) / (OGD * OG);
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, OL * OG, n);
    ggml_set_op_params_i32(result, 0, 1);
    result->op     = GGML_OP_DSV41_OPROJ;
    result->src[0] = o;
    result->src[1] = wo_a;
    return result;
}

static void ggml_dsv41_set_rope(struct ggml_tensor * t, int i0, struct ggml_dsv41_rope r) {""")], "ggml_dsv41_oproj_a")

SC = os.path.join(G, "src/ggml-sycl/ggml-sycl.cpp")
edit(SC, [
# the batched product as a helper (patch-slicesbatch.py's block), then larry_slices_gemm's call of it
("""// prompt chunks against a per-slice planar Q6_K weight: per slice, W (f16) and the slice's columns (f16), one GEMM
static bool larry_slices_gemm(""",
"""// larry, patch-slicesbatch.py / patch-oproja.py: dst_g [ncols][N] (ld ldc, slices sc apart) = X_g [ncols][K] . W_g^T
// for every slice in ONE gemm_batch - xs [ncols][G][K] f32 contiguous (converted to fp16 once), the per-slice planar
// Q6_K slices (w0, sb bytes apart) dequantised into one fp16 cache entry
static void larry_slices_batched(ggml_backend_sycl_context & ctx, const char * w0, size_t sb, int64_t N, int64_t K, int64_t G,
                                 const float * xs, int64_t ncols, float * dd, int64_t ldc, int64_t sc) {
    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> xa(ctx.pool(), (size_t) (ncols * G * K));
    ggml_sycl_pool_alloc<sycl::half> wa(ctx.pool());   // only when the fp16 cache cannot hold the whole weight
    auto deq = [&](sycl::half * y) {
        for (int64_t g = 0; g < G; ++g) fastmv::dequant_q6k_planar_f16(*st, w0 + g * sb, N, K, y + g * N * K);
    };
    const sycl::half * w = larry_f16cache_core(w0, (size_t) (N * K * G), deq, st);
    if (!w) {
        wa.alloc((size_t) (N * K * G));
        deq(wa.get());
        w = wa.get();
    }
    sycl::half * xh = xa.get();
    const size_t ne4 = (size_t) (ncols * G * K) / 4;
    st->parallel_for(sycl::range<1>(ne4), [=](sycl::id<1> i) {
        const sycl::float4 v = *(const sycl::float4 *) (xs + 4 * i[0]);
        *(sycl::vec<sycl::half, 4> *) (xh + 4 * i[0]) =
            sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
    });
    oneapi::mkl::blas::row_major::gemm_batch(*st, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, ncols, N, K,
                                             1.0f, xh, G * K, K, w, K, N * K, 0.0f, dd, ldc, sc, G);
}

// prompt chunks against a per-slice planar Q6_K weight: per slice, W (f16) and the slice's columns (f16), one GEMM
static bool larry_slices_gemm("""),
("""        dst->nb[1] % 4 == 0 && dst->nb[2] % 4 == 0) {
        ggml_sycl_pool_alloc<sycl::half> xa(ctx.pool(), (size_t) (ncols * G * K));
        ggml_sycl_pool_alloc<sycl::half> wa(ctx.pool());   // only when the fp16 cache cannot hold the whole weight
        const char * w0 = (const char *) src0->data;
        const size_t sb = src0->nb[2];
        auto deq = [&](sycl::half * y) {
            for (int64_t g = 0; g < G; ++g) fastmv::dequant_q6k_planar_f16(*st, w0 + g * sb, N, K, y + g * N * K);
        };
        const sycl::half * w = larry_f16cache_core(w0, (size_t) (N * K * G), deq, st);
        if (!w) {
            wa.alloc((size_t) (N * K * G));
            deq(wa.get());
            w = wa.get();
        }
        const float * xs = (const float *) src1->data;
        sycl::half * xh = xa.get();
        const size_t ne4 = (size_t) (ncols * G * K) / 4;
        st->parallel_for(sycl::range<1>(ne4), [=](sycl::id<1> i) {
            const sycl::float4 v = *(const sycl::float4 *) (xs + 4 * i[0]);
            *(sycl::vec<sycl::half, 4> *) (xh + 4 * i[0]) =
                sycl::vec<sycl::half, 4>((sycl::half) v.x(), (sycl::half) v.y(), (sycl::half) v.z(), (sycl::half) v.w());
        });
        // dst slice g, row-major [ncols][N] (ld nb1 / 4, slices nb2 / 4 apart) = X_g [ncols x K] (ld G K) . W_g^T
        oneapi::mkl::blas::row_major::gemm_batch(*st, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, ncols, N, K,
                                                 1.0f, xh, G * K, K, w, K, N * K, 0.0f, (float *) dst->data,
                                                 (int64_t) (dst->nb[1] / 4), (int64_t) (dst->nb[2] / 4), G);
        return true;
    }""",
"""        dst->nb[1] % 4 == 0 && dst->nb[2] % 4 == 0) {
        // dst slice g, row-major [ncols][N] (ld nb1 / 4, slices nb2 / 4 apart)
        larry_slices_batched(ctx, (const char *) src0->data, src0->nb[2], N, K, G, (const float *) src1->data, ncols,
                             (float *) dst->data, (int64_t) (dst->nb[1] / 4), (int64_t) (dst->nb[2] / 4));
        return true;
    }"""),
# the op: wo_a alone in wo_b's layout
("""static bool larry_oproj(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {""",
"""// larry, patch-oproja.py: GGML_OP_DSV41_OPROJ with op_params[0] = 1 (ggml_dsv41_oproj_a) - wo_a at prompt widths, its
// slices in one gemm_batch, written in wo_b's input layout: column t, group g at t * OL * OG + g * OL
static bool larry_oproj_a(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * o = dst->src[0];
    const ggml_tensor * wa = dst->src[1];   // the 3-D view of the 2-D weight
    const int64_t K = wa->ne[0], N = wa->ne[1], G = wa->ne[2], n = dst->ne[1];
    if (wa->type != GGML_TYPE_Q6_K || K % QK_K || (n * G * K) % 4 || dst->ne[0] != N * G) return false;
    if (!g_larry_slice_planar.count(wa->data)) {   // per slice, once (as larry_fastmv's wo_a branch)
        const ggml_tensor * par = wa->view_src ? wa->view_src : wa;
        if (!reorder_qw_q6_k_moe((uint8_t *) par->data, wa->nb[2], G, ctx.stream())) return false;
        g_larry_slice_planar.insert(wa->data);
    }
    larry_slices_batched(ctx, (const char *) wa->data, wa->nb[2], N, K, G, (const float *) o->data, n, (float *) dst->data,
                         N * G, N);
    return true;
}

static bool larry_oproj(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {"""),
("""        case GGML_OP_DSV41_OPROJ:
            if (!larry_oproj(ctx, dst)) GGML_ABORT("DSV41_OPROJ: shape or layout not covered");""",
"""        case GGML_OP_DSV41_OPROJ:   // (op_params[0] = 1: ggml_dsv41_oproj_a, patch-oproja.py)
            if (!(ggml_get_op_params_i32(dst, 0) == 1 ? larry_oproj_a(ctx, dst) : larry_oproj(ctx, dst)))
                GGML_ABORT("DSV41_OPROJ: shape or layout not covered");"""),
("""        case GGML_OP_DSV41_OPROJ:
            return op->src[1]->type == GGML_TYPE_Q6_K && op->src[2]->type == GGML_TYPE_Q6_K && op->ne[1] <= 8;""",
"""        case GGML_OP_DSV41_OPROJ:
            if (ggml_get_op_params_i32(op, 0) == 1) return op->src[1]->type == GGML_TYPE_Q6_K;   // patch-oproja.py
            return op->src[1]->type == GGML_TYPE_Q6_K && op->src[2]->type == GGML_TYPE_Q6_K && op->ne[1] <= 8;"""),
], "larry_oproj_a")

edit(os.path.join(X, "tools/v41/engine/ds41_dense.cpp"), [(
"""    } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
    }""",
"""    } else {
        // larry, STRATA_OPROJ_A (patch-oproja.py, default 0): prompt widths take wo_a as one op that writes wo_b's input
        // layout (the mul_mat's [OL, n, OG] was permuted and copied back: 0.16 ms a 512-column span)
        static const bool oproj_a = env_on("STRATA_OPROJ_A", false);
        if (oproj_a && im.sycl && n > 8 && ggml_is_contiguous(o) && wo_a->type == GGML_TYPE_Q6_K) {
            attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), ggml_dsv41_oproj_a(gc, o, wo_a));   // [D, n]
        } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
        }
    }""")], "STRATA_OPROJ_A")
