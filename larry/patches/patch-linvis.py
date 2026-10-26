#!/usr/bin/env python3
# patch-linvis.py [XOLOTL_DIR] - larry, STRATA_LI_NVIS (default 0): the prompt indexer told each query's visible row
# count instead of a [cap, n] 0 / -inf mask.  The prompt path built that mask per call - log(step(repeat(ramp) + nvis)),
# then an F16 copy (REPEAT 143 + ADD ~90 + STEP 138 + LOG 130 + CPY 362 us a call at a 31K prompt's second batch,
# abwT110) - and read it again for the selected rows' mask (GET_ROWS).  ggml_lightning_indexer_nvis = the same op with
# nvis F32 [n] as src[3] (op_params[0] = 1; ggml-sycl's li_fused / li_gemm take it, the CPU op asserts on it); the
# selected rows' mask is then log(step(nvis - sel)) on [K, n].  Needs patch-lifused.py first.  Idempotent.
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


# ---- ggml.h / ggml.c: the constructor
edit(os.path.join(G, "include/ggml.h"), [(
"""    GGML_API struct ggml_tensor * ggml_lightning_indexer(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * mask);
""",
"""    GGML_API struct ggml_tensor * ggml_lightning_indexer(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * mask);

    // larry: the same op with each query's visible row count in place of the mask - nvis F32 [n_batch]: row i of k is
    // visible to query t iff i < nvis[t] (the result -inf elsewhere, as the 0 / -inf mask gives).  ggml-sycl only
    // (op_params[0] = 1; the CPU op asserts on the F32 src[3]).
    GGML_API struct ggml_tensor * ggml_lightning_indexer_nvis(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis);
""")], "ggml_lightning_indexer_nvis")

edit(os.path.join(G, "src/ggml.c"), [(
"""    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    return result;
}
""",
"""    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    return result;
}

// larry: ggml_lightning_indexer with nvis F32 [n_batch] (rows i < nvis[t] visible to query t) for the mask
struct ggml_tensor * ggml_lightning_indexer_nvis(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * nvis) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    nvis->type == GGML_TYPE_F32);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(ggml_nelements(nvis) == q->ne[2]);
    GGML_ASSERT(ggml_is_contiguous(nvis));
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && weights->ne[3] == 1);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[1] = { 1 };
    ggml_set_op_params(result, params, sizeof(params));

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = nvis;

    return result;
}
""")], "ggml_lightning_indexer_nvis")

# ---- ggml-sycl: supports_op, and the two prompt paths read nvis (the decode XMX path and the generic kernel never get it)
edit(os.path.join(G, "src/ggml-sycl/ggml-sycl.cpp"), [(
"""                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;""",
"""                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3]->type == GGML_TYPE_F16 ||   // larry: or nvis F32 [n] (ggml_lightning_indexer_nvis)
                 (op->op_params[0] == 1 && op->src[3]->type == GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_Q8_0 &&
                  op->src[1]->type != GGML_TYPE_Q5_1 && op->src[1]->type != GGML_TYPE_Q5_0 && op->src[1]->type != GGML_TYPE_Q4_1 &&
                  op->src[1]->type != GGML_TYPE_Q4_0 && op->src[1]->type != GGML_TYPE_IQ4_NL)) &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;""")], "ggml_lightning_indexer_nvis")

LI = os.path.join(G, "src/ggml-sycl/lightning-indexer.cpp")
edit(LI, [
# li_gemm's head-sum kernel: the mask or nvis
("""    const char * wd = (const char *) w->data;
    const char * md = (const char *) m->data;
    float * od = (float *) dst->data;
    for (int64_t j0 = 0; j0 < NKV; j0 += CH) {
        const int64_t nc = std::min<int64_t>(CH, NKV - j0);
        mkl::blas::row_major::gemm(*stream, mkl::transpose::nontrans, mkl::transpose::trans, MR, nc, E, 1.0f, qh, E,
                                   kh + j0 * E, E, 0.0f, s, nc);
        stream->parallel_for(sycl::range<2>((size_t) B, (size_t) nc), [=](sycl::id<2> id) {
            const int64_t b = (int64_t) id[0], jj = (int64_t) id[1];
            const float * wb = (const float *) (wd + b * nbw1);
            float acc = 0.0f;
            for (int64_t h = 0; h < H; ++h) acc += sycl::fmax(s[(b * H + h) * nc + jj], 0.0f) * wb[h];
            const sycl::half * mb = (const sycl::half *) (md + b * nbm1);
            od[(j0 + jj) + b * (int64_t) (nb1 / sizeof(float))] = acc + (float) mb[j0 + jj];
        });
    }""",
"""    const char * wd = (const char *) w->data;
    const char * md = (const char *) m->data;
    float * od = (float *) dst->data;
    const bool nvm = dst->op_params[0] == 1;   // larry, patch-linvis.py: src[3] = nvis F32 [B]
    for (int64_t j0 = 0; j0 < NKV; j0 += CH) {
        const int64_t nc = std::min<int64_t>(CH, NKV - j0);
        mkl::blas::row_major::gemm(*stream, mkl::transpose::nontrans, mkl::transpose::trans, MR, nc, E, 1.0f, qh, E,
                                   kh + j0 * E, E, 0.0f, s, nc);
        stream->parallel_for(sycl::range<2>((size_t) B, (size_t) nc), [=](sycl::id<2> id) {
            const int64_t b = (int64_t) id[0], jj = (int64_t) id[1];
            const float * wb = (const float *) (wd + b * nbw1);
            float acc = 0.0f;
            for (int64_t h = 0; h < H; ++h) acc += sycl::fmax(s[(b * H + h) * nc + jj], 0.0f) * wb[h];
            const float mv = nvm ? ((float) (j0 + jj) < ((const float *) md)[b] ? 0.0f : -INFINITY)
                                 : (float) ((const sycl::half *) (md + b * nbm1))[j0 + jj];
            od[(j0 + jj) + b * (int64_t) (nb1 / sizeof(float))] = acc + mv;
        });
    }"""),
# the decode XMX path: mask only
("""    if (!on) return false;
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    return k->type == GGML_TYPE_BF16 && q->type == GGML_TYPE_F32 && q->ne[0] == 128 && q->ne[1] == 32 && q->ne[2] >= 1 &&""",
"""    if (!on || dst->op_params[0] == 1) return false;   // (nvis: patch-linvis.py)
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    return k->type == GGML_TYPE_BF16 && q->type == GGML_TYPE_F32 && q->ne[0] == 128 && q->ne[1] == 32 && q->ne[2] >= 1 &&"""),
# li_fused: accept nvis
("""    if (q->ne[0] != 128 || q->ne[1] != 32 || q->ne[2] < min_b || q->ne[3] != 1 || k->ne[0] != 128 || k->ne[1] != 1 ||
        k->ne[3] != 1 || m->ne[3] != 1 || q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        m->nb[0] != sizeof(sycl::half) || w->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (w->nb[1] % sizeof(float) || m->nb[1] % sizeof(sycl::half) || dst->nb[1] % sizeof(float)) return false;""",
"""    if (q->ne[0] != 128 || q->ne[1] != 32 || q->ne[2] < min_b || q->ne[3] != 1 || k->ne[0] != 128 || k->ne[1] != 1 ||
        k->ne[3] != 1 || q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        w->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (w->nb[1] % sizeof(float) || dst->nb[1] % sizeof(float)) return false;
    if (dst->op_params[0] == 1) return m->type == GGML_TYPE_F32 && ggml_is_contiguous(m);   // nvis (patch-linvis.py)
    if (m->ne[3] != 1 || m->nb[0] != sizeof(sycl::half) || m->nb[1] % sizeof(sycl::half)) return false;"""),
("""static void lif_run(const sycl::half * qh, const sycl::half * kh, int64_t NKV, const float * w, int64_t ldw,
                    const sycl::half * m, int64_t ldm, float * od, int64_t ldo, int64_t B, int QW, int64_t KR, queue_ptr stream) {""",
"""static void lif_run(const sycl::half * qh, const sycl::half * kh, int64_t NKV, const float * w, int64_t ldw,
                    const sycl::half * m, int64_t ldm, const float * nvis, float * od, int64_t ldo, int64_t B, int QW, int64_t KR,
                    queue_ptr stream) {"""),
("""        const sycl::half * mr = m + b * ldm;
        float * orow = od + b * ldo;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja < NKV ? (float) mr[ja] : -INFINITY;
            const float mb = jb < NKV ? (float) mr[jb] : -INFINITY;""",
"""        const sycl::half * mr = m + b * ldm;
        const float nv = nvis ? nvis[b] : 0.0f;   // nvis: rows j < nv visible (patch-linvis.py)
        float * orow = od + b * ldo;
        const int64_t j1 = sycl::min(NKV, (kr + 1) * KR);
        for (int64_t j0 = kr * KR; j0 < j1; j0 += 32) {
            const int64_t ja = j0 + lane, jb = j0 + 16 + lane;
            const float ma = ja >= NKV ? -INFINITY : nvis ? ((float) ja < nv ? 0.0f : -INFINITY) : (float) mr[ja];
            const float mb = jb >= NKV ? -INFINITY : nvis ? ((float) jb < nv ? 0.0f : -INFINITY) : (float) mr[jb];"""),
("""    lif_run(qh, kh, NKV, (const float *) w->data, (int64_t) (w->nb[1] / sizeof(float)), (const sycl::half *) m->data,
            (int64_t) (m->nb[1] / sizeof(sycl::half)), (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)), B, qw, KR, stream);""",
"""    const bool nvm = dst->op_params[0] == 1;
    lif_run(qh, kh, NKV, (const float *) w->data, (int64_t) (w->nb[1] / sizeof(float)), nvm ? nullptr : (const sycl::half *) m->data,
            nvm ? 0 : (int64_t) (m->nb[1] / sizeof(sycl::half)), nvm ? (const float *) m->data : nullptr, (float *) dst->data,
            (int64_t) (dst->nb[1] / sizeof(float)), B, qw, KR, stream);"""),
# the generic per-row kernel: mask only (prompt widths reach li_fused / li_gemm)
("""    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);""",
"""    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);   // (an nvis node - patch-linvis.py - must reach li_fused or li_gemm)"""),
], "patch-linvis.py")

# ---- the engine's prompt path
edit(os.path.join(X, "tools/v41/engine/ds41_dense.cpp"), [
("""                    // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                    ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, (size_t) qc0 * 4), 1, nq);
                    ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, nq)), nvs)));
                    ggml_tensor * sc = ggml_lightning_indexer(gc, iqs, ksrc, iws, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                    sc = ggml_reshape_2d(gc, sc, cap, nq);""",
"""                    // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                    ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, (size_t) qc0 * 4), 1, nq);
                    // larry, STRATA_LI_NVIS (patch-linvis.py, default 0): ggml-sycl's indexer takes nvis itself - no [cap, nq]
                    // mask (repeat + add + step + log + F16 copy a call), the selected rows' mask from their ids below
                    static const bool li_nvis = env_on("STRATA_LI_NVIS", false);
                    ggml_tensor * vis = nullptr, * sc = nullptr;
                    if (li_nvis && im.sycl) {
                        sc = ggml_lightning_indexer_nvis(gc, iqs, ksrc, iws, nvs);   // [cap, nq], -inf past nvis
                    } else {
                        vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, nq)), nvs)));
                        sc = ggml_lightning_indexer(gc, iqs, ksrc, iws, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                    }
                    sc = ggml_reshape_2d(gc, sc, cap, nq);"""),
("""                    selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, nq), sel), K, nq);   // vis at sel""",
"""                    if (vis) {
                        selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, nq), sel), K, nq);   // vis at sel
                    } else {   // 0 where sel < nvis, else -inf: log(step(nvis - sel)) on [K, nq]
                        selm = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_scale(gc, ggml_cast(gc, sel, GGML_TYPE_F32), -1.0f), nvs)));
                    }"""),
], "STRATA_LI_NVIS")
