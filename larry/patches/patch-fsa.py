#!/usr/bin/env python3
# patch-fsa.py [XOLOTL_DIR] - larry, 2026-10-28 (#3, big prompts): the prompt attention of a span as ONE kernel that reads
# each query's keys where they lie (patches/fsa_kernel.inc, embedded in ggml-sycl's fattn-dsv4.cpp).
#  - ggml.h / ggml.c: ggml_dsv41_attn_fsa(q, win, widx, comp, sel, mask, sinks, scale, swa) - a GGML_OP_DSV41_ATTN_SDPA
#    node with op_params {scale, M, 2 (the fused form), swa}, sources q / win / mask / sinks / widx / comp / sel, an F16
#    [DH, NH, n] result (as ggml_dsv41_attn_sdpa_o16's).
#  - ggml-sycl: the op runs fsa::run (the window rows in fp16 by position, then the attention: 64 groups, 12 MiB of pool
#    scratch + the window copy); supports_op for the form.
#  - ds41_dense.cpp, STRATA_ATTN_FSA=1 (default 0; with STRATA_ATTN_SDPA and its fp16 output, STRATA_SDPA_O16): a span
#    whose keys are its 128 window rows (+ 512 BF16 compressed rows) takes the fused op instead of DSV41_KGATHER (387 MB
#    written a 512-column span) + the SDPA (the Q' / mask kernels, oneDNN's graph SDPA, the output copy).
# bench/fa/test_fsa.cpp (abwT156): 512 queries at 30K 1.44 ms against 2.37, at 130K 1.84 against 2.64; within the
# oneDNN path's error of a double reference.  GGML_SYCL_FSA_SLOTS (64): the work-groups.  Backups of the ggml files:
# /opt/ai/dsv41/logs/ggml-backups/*.pre-fsa-20261028.  Idempotent.
import os
import shutil
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
G = os.path.join(X, "third_party/llama.cpp/ggml")
INC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fsa_kernel.inc")
BK = "/opt/ai/dsv41/logs/ggml-backups"


def edit(path, pairs, marker, backup=True):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:90]))
        s = s.replace(old, new)
    if backup and os.path.isdir(BK):
        b = os.path.join(BK, os.path.basename(path) + ".pre-fsa-20261028")
        if not os.path.exists(b):
            shutil.copy2(path, b)
    open(path, "w").write(s)
    print("patched", path)


edit(os.path.join(G, "include/ggml.h"), [(
"""    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa(""",
"""    // larry (patch-fsa.py): the prompt attention as one kernel that reads each query's keys where they lie - its swa
    // window rows (rows of win, widx: I32 [swa * n]) then its selected compressed rows (BF16 rows of comp at sel: I32
    // [K, n]; comp NULL: the window only), mask: F32 [M, n] (0 / -inf) over the M = swa (+ K) keys.  F16 [DH, NH, n].
    GGML_API struct ggml_tensor * ggml_dsv41_attn_fsa(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * win,
            struct ggml_tensor  * widx,
            struct ggml_tensor  * comp,
            struct ggml_tensor  * sel,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * sinks,
            float                 scale,
            int                   swa);
    GGML_API struct ggml_tensor * ggml_dsv41_attn_sdpa(""")], "ggml_dsv41_attn_fsa")

edit(os.path.join(G, "src/ggml.c"), [(
"""// larry, patch-sdpao16.py: ggml_dsv41_attn_sdpa with an F16 result""",
"""// larry, patch-fsa.py: the fused prompt attention (a GGML_OP_DSV41_ATTN_SDPA node of form 2: op_params {scale, M, 2, swa};
// sources q, win, mask, sinks, widx, comp, sel)
struct ggml_tensor * ggml_dsv41_attn_fsa(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * win,
        struct ggml_tensor  * widx,
        struct ggml_tensor  * comp,
        struct ggml_tensor  * sel,
        struct ggml_tensor  * mask,
        struct ggml_tensor  * sinks,
        float                 scale,
        int                   swa) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && q->ne[3] == 1);
    const int64_t DH = q->ne[0], NH = q->ne[1], n = q->ne[2];
    GGML_ASSERT(win->type == GGML_TYPE_F32 && win->ne[0] == DH && win->nb[0] == 4);
    GGML_ASSERT(widx->type == GGML_TYPE_I32 && ggml_is_contiguous(widx) && ggml_nelements(widx) == (int64_t) swa * n);
    GGML_ASSERT((comp == NULL) == (sel == NULL));
    const int64_t M = mask->ne[0];
    GGML_ASSERT(mask->type == GGML_TYPE_F32 && mask->nb[0] == 4 && ggml_nrows(mask) == n);
    if (comp) {
        GGML_ASSERT(comp->type == GGML_TYPE_BF16 && comp->ne[0] == DH && comp->nb[0] == 2);
        GGML_ASSERT(sel->type == GGML_TYPE_I32 && sel->nb[0] == 4 && sel->ne[1] == n && sel->ne[0] >= M - swa);
    } else {
        GGML_ASSERT(M == swa);
    }
    GGML_ASSERT(sinks && sinks->type == GGML_TYPE_F32 && ggml_nelements(sinks) == NH);
    struct ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, DH, NH, n);
    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, (int32_t) M);
    ggml_set_op_params_i32(result, 2, 2);
    ggml_set_op_params_i32(result, 3, swa);
    result->op     = GGML_OP_DSV41_ATTN_SDPA;
    result->src[0] = q;
    result->src[1] = win;
    result->src[2] = mask;
    result->src[3] = sinks;
    result->src[4] = widx;
    result->src[5] = comp;
    result->src[6] = sel;
    return result;
}

// larry, patch-sdpao16.py: ggml_dsv41_attn_sdpa with an F16 result""")], "ggml_dsv41_attn_fsa")

kernel = open(INC).read()
edit(os.path.join(G, "src/ggml-sycl/fattn-dsv4.cpp"), [
("""void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    namespace mkl = oneapi::mkl;""",
"""// ---- larry, patch-fsa.py: the fused prompt attention (GGML_OP_DSV41_ATTN_SDPA form 2, ggml_dsv41_attn_fsa)
""" + kernel + """
static void ggml_sycl_op_dsv41_attn_fsa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * W = dst->src[1], * MK = dst->src[2], * SK = dst->src[3];
    const ggml_tensor * WI = dst->src[4], * CM = dst->src[5], * SL = dst->src[6];
    const int64_t n = Q->ne[2];
    GGML_ASSERT(Q->ne[0] == fsa::DH && Q->ne[1] == fsa::NH && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    static const int slots = [] { const char * v = std::getenv("GGML_SYCL_FSA_SLOTS"); return v && *v ? std::max(1, std::atoi(v)) : 64; }();
    fsa::Args a;
    memcpy(&a.scale, (const float *) dst->op_params + 0, sizeof(float));
    a.M = ggml_get_op_params_i32(dst, 1);
    a.swa = ggml_get_op_params_i32(dst, 3);
    a.q = (const float *) Q->data;
    a.win = (const char *) W->data;
    a.wnb = W->nb[1];
    a.widx = (const int32_t *) WI->data;
    a.comp = CM ? (const char *) CM->data : nullptr;
    a.cnb = CM ? CM->nb[1] : 0;
    a.sel = SL ? (const int32_t *) SL->data : nullptr;
    a.ldsel = SL ? (int) (SL->nb[1] / 4) : 0;
    a.mask = (const float *) MK->data;
    a.ldm = (int) (MK->nb[1] / 4);
    a.sinks = (const float *) SK->data;
    a.out = (sycl::half *) dst->data;
    a.n = (int) n;
    a.nslot = (int) std::min<int64_t>(slots, n);
    GGML_ASSERT(a.M % fsa::CK == 0 && a.swa == fsa::CK);
    ggml_sycl_pool & pool = ctx.pool();
    ggml_sycl_pool_alloc<sycl::half> scr(pool, (size_t) a.nslot * fsa::SLOT);
    ggml_sycl_pool_alloc<sycl::half> wcp(pool, (size_t) (n + a.swa) * fsa::DH);
    a.scratch = scr.get();
    a.wcopy = wcp.get();
    fsa::run(*ctx.stream(), a);
}

void ggml_sycl_op_dsv41_attn_sdpa(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    namespace mkl = oneapi::mkl;
    if (ggml_get_op_params_i32(dst, 2) == 2) { ggml_sycl_op_dsv41_attn_fsa(ctx, dst); return; }   // larry, patch-fsa.py"""),
], "patch-fsa.py")

edit(os.path.join(G, "src/ggml-sycl/ggml-sycl.cpp"), [(
"""        case GGML_OP_DSV41_ATTN_SDPA:   // larry, patch-sdpa
            return op->src[0]->ne[0] % 4 == 0 && op->src[1]->ne[0] % 16 == 0;""",
"""        case GGML_OP_DSV41_ATTN_SDPA:   // larry, patch-sdpa
            if (op->op_params[2] == 2)   // larry, patch-fsa.py: the fused form (fsa_kernel.inc's shapes)
                return op->type == GGML_TYPE_F16 && op->src[0]->ne[0] == 512 && op->src[0]->ne[1] == 64 && op->op_params[3] == 128 &&
                       op->op_params[1] % 128 == 0 && op->src[1]->nb[1] % 16 == 0 && (!op->src[5] || op->src[5]->nb[1] % 16 == 0);
            return op->src[0]->ne[0] % 4 == 0 && op->src[1]->ne[0] % 16 == 0;""")], "patch-fsa.py")

D = os.path.join(X, "tools/v41/engine/ds41_dense.cpp")
edit(D, [(
"""    if (attn_sdpa && sycl_mqa && DH == 512 && (M == SWA || (Kc && M == SWA + 512))) {
        ggml_tensor * kp = ggml_dsv41_kgather_pad(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, sdpa_dp);
        // larry, STRATA_SDPA_O16 (patch-sdpao16.py, default 0): the output in fp16 when wo_a takes it as one op (STRATA_OPROJ_A)
        static const bool sdpa_o16 = env_on("STRATA_SDPA_O16", false) && env_on("STRATA_OPROJ_A", false);
        const bool o16 = sdpa_o16 && im.sycl && n > 8 && im.BL(il, "attn_output_a.weight")->type == GGML_TYPE_Q6_K;
        o = (o16 ? ggml_dsv41_attn_sdpa_o16 : ggml_dsv41_attn_sdpa)(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"),
                                                                   1.0f / std::sqrt((float) DH));   // [DH, NH, n]""",
"""    if (attn_sdpa && sycl_mqa && DH == 512 && (M == SWA || (Kc && M == SWA + 512))) {
        // larry, STRATA_SDPA_O16 (patch-sdpao16.py, default 0): the output in fp16 when wo_a takes it as one op (STRATA_OPROJ_A)
        static const bool sdpa_o16 = env_on("STRATA_SDPA_O16", false) && env_on("STRATA_OPROJ_A", false);
        const bool o16 = sdpa_o16 && im.sycl && n > 8 && im.BL(il, "attn_output_a.weight")->type == GGML_TYPE_Q6_K;
        // larry, STRATA_ATTN_FSA (patch-fsa.py, default 0; with the fp16 output): one kernel that reads each query's window
        // rows and selected compressed rows where they lie - no [n, 656, 576] gathered copy, no Q' / mask / output kernels
        static const bool attn_fsa = env_on("STRATA_ATTN_FSA", false);
        if (attn_fsa && o16 && SWA == 128 && NH == 64 && (!Kc || kg_comp->type == GGML_TYPE_BF16) && ggml_is_contiguous(Mall)) {
            o = ggml_dsv41_attn_fsa(gc, q, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, Mall,
                                    im.BL(il, "attn_sinks.weight"), 1.0f / std::sqrt((float) DH), (int) SWA);   // F16 [DH, NH, n]
        } else {
        ggml_tensor * kp = ggml_dsv41_kgather_pad(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, sdpa_dp);
        o = (o16 ? ggml_dsv41_attn_sdpa_o16 : ggml_dsv41_attn_sdpa)(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"),
                                                                   1.0f / std::sqrt((float) DH));   // [DH, NH, n]
        }   // (patch-fsa.py)""")], "STRATA_ATTN_FSA", backup=False)
