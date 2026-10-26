#!/usr/bin/env python3
# patch-tpoproj.py - larry, STRATA_TP_OPROJ (default 0 for the A/B): DeepSeek-V4.1's attention output (wo_a's 8 head
# groups, then wo_b) at decode / verify widths as one ggml op, GGML_OP_DSV41_OPROJ, that ggml-sycl hands to the xe tier:
# card k takes groups 2k, 2k+1 and wo_b's matching 2048 columns, sends its partial sum (5120 floats a column) to card 0,
# card 0 computes its own quarter and adds the parts.  bench/tp/tp_probe: 131.6 -> 45.6 us on the GPUs (the host's 22
# submissions ~93 us - decode's host has ~230 us of readback wait a layer to hide them in).  Idempotent.
import sys

ROOT = "/opt/ai/dsv41/xolotl"
GG = ROOT + "/third_party/llama.cpp/ggml"
FILES = {
    "h": GG + "/include/ggml.h",
    "c": GG + "/src/ggml.c",
    "cpu": GG + "/src/ggml-cpu/ggml-cpu.c",
    "sh": GG + "/include/ggml-sycl.h",
    "sc": GG + "/src/ggml-sycl/ggml-sycl.cpp",
    "dense": ROOT + "/tools/v41/engine/ds41_dense.cpp",
    "th": ROOT + "/tools/v41/engine/xe/ds41_xe_tier.hpp",
    "tc": ROOT + "/tools/v41/engine/xe/ds41_xe_tier.cpp",
    "gen": ROOT + "/tools/v41/engine/ds41_generate.cpp",
}
src = {k: open(p).read() for k, p in FILES.items()}
if "GGML_OP_DSV41_OPROJ" in src["h"]:
    print("already"); sys.exit(0)


def rep(k, old, new, cnt=1):
    s = src[k]
    if s.count(old) != cnt:
        sys.exit(f"{k}: anchor x{s.count(old)}: {old[:100]!r}")
    src[k] = s.replace(old, new)


# ---------------------------------------------------------------- ggml.h
rep("h", """        GGML_OP_DSV41_HC_FB,

        GGML_OP_UNARY,""", """        GGML_OP_DSV41_HC_FB,
        GGML_OP_DSV41_OPROJ,

        GGML_OP_UNARY,""")
rep("h", """    GGML_API struct ggml_tensor * ggml_dsv41_swiglu_lim(
            struct ggml_context * ctx,
            struct ggml_tensor  * y,
            float                 lim);
""", """    GGML_API struct ggml_tensor * ggml_dsv41_swiglu_lim(
            struct ggml_context * ctx,
            struct ggml_tensor  * y,
            float                 lim);

    // larry: DeepSeek-V4.1's attention output at decode widths as one op: dst[D, n] = wo_b @ concat_g(wo_a[g] @ o[g]),
    // o: f32 contiguous, OGD * OG values a column (a group's heads adjacent); wo_a: the 3-D view [OGD, OL, OG]; wo_b:
    // [OL * OG, D].  ggml-sycl only (an external launcher may take it: ggml_backend_sycl_set_oproj_hook).
    GGML_API struct ggml_tensor * ggml_dsv41_oproj(
            struct ggml_context * ctx,
            struct ggml_tensor  * o,
            struct ggml_tensor  * wo_a,
            struct ggml_tensor  * wo_b);
""")

# ---------------------------------------------------------------- ggml.c
rep("c", """    "DSV41_HC_FB",

    "UNARY",""", """    "DSV41_HC_FB",
    "DSV41_OPROJ",

    "UNARY",""")
rep("c", """    "dsv41_hc_fb(fa, scale, base, w)",

    "unary(x)",""", """    "dsv41_hc_fb(fa, scale, base, w)",
    "dsv41_oproj(o, wo_a, wo_b)",

    "unary(x)",""")
rep("c", 'static_assert(GGML_OP_COUNT == 110, "GGML_OP_COUNT != 110");', 'static_assert(GGML_OP_COUNT == 111, "GGML_OP_COUNT != 111");', 2)
rep("c", """    result->op     = GGML_OP_DSV41_SWIGLU_LIM;
    result->src[0] = y;
    return result;
}
""", """    result->op     = GGML_OP_DSV41_SWIGLU_LIM;
    result->src[0] = y;
    return result;
}

struct ggml_tensor * ggml_dsv41_oproj(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a,
        struct ggml_tensor  * wo_b) {
    GGML_ASSERT(o->type == GGML_TYPE_F32 && ggml_is_contiguous(o));
    const int64_t OGD = wo_a->ne[0], OL = wo_a->ne[1], OG = wo_a->ne[2];
    GGML_ASSERT(wo_b->ne[0] == OL * OG && ggml_nelements(o) % (OGD * OG) == 0);
    const int64_t n = ggml_nelements(o) / (OGD * OG);
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, wo_b->ne[1], n);
    result->op     = GGML_OP_DSV41_OPROJ;
    result->src[0] = o;
    result->src[1] = wo_a;
    result->src[2] = wo_b;
    return result;
}
""")

# ---------------------------------------------------------------- ggml-cpu.c
rep("cpu", """        case GGML_OP_DSV41_SWIGLU_LIM:
            {
                ggml_compute_forward_dsv41_swiglu_lim(params, tensor);
            } break;
""", """        case GGML_OP_DSV41_SWIGLU_LIM:
            {
                ggml_compute_forward_dsv41_swiglu_lim(params, tensor);
            } break;
        case GGML_OP_DSV41_OPROJ:
            {
                GGML_ABORT("DSV41_OPROJ is a ggml-sycl op (no CPU implementation)");
            } break;
""")
rep("cpu", """        case GGML_OP_DSV41_HC_FA:
        case GGML_OP_DSV41_HC_FB:
            {
                n_tasks = 1;
            } break;""", """        case GGML_OP_DSV41_HC_FA:
        case GGML_OP_DSV41_HC_FB:
        case GGML_OP_DSV41_OPROJ:
            {
                n_tasks = 1;
            } break;""")

# ---------------------------------------------------------------- ggml-sycl.h
rep("sh", """GGML_BACKEND_API void ggml_backend_sycl_set_mv_hook(ggml_sycl_larry_mv_hook_t fn, void * user, double min_mb);
""", """GGML_BACKEND_API void ggml_backend_sycl_set_mv_hook(ggml_sycl_larry_mv_hook_t fn, void * user, double min_mb);
// larry: an external launcher for GGML_OP_DSV41_OPROJ (DeepSeek-V4.1's attention output, <= 8 columns): the backend's
// queue, wo_a's per-slice planar Q6_K (OG slices of OL rows x OGD, slice stride in bytes), wo_b's planar Q6_K (D rows
// x OL * OG), o (f32, ncols columns of OGD * OG, column stride in floats) and dst (column stride in floats).  True if it
// queued the whole product, ordered on that queue.
typedef bool (*ggml_sycl_larry_oproj_hook_t)(void * queue, const void * wo_a, size_t wo_a_slice_bytes, const void * wo_b,
                                             int64_t OGD, int64_t OL, int64_t OG, int64_t D, int ncols, const float * o,
                                             int64_t o_col_stride, float * dst, int64_t dst_col_stride, void * user);
GGML_BACKEND_API void ggml_backend_sycl_set_oproj_hook(ggml_sycl_larry_oproj_hook_t fn, void * user);
""")

# ---------------------------------------------------------------- ggml-sycl.cpp
rep("sc", """void ggml_backend_sycl_set_mv_hook(ggml_sycl_larry_mv_hook_t fn, void * user, double min_mb) {
    g_larry_mv_hook = fn; g_larry_mv_user = user; g_larry_mv_mb = min_mb;
}
""", """void ggml_backend_sycl_set_mv_hook(ggml_sycl_larry_mv_hook_t fn, void * user, double min_mb) {
    g_larry_mv_hook = fn; g_larry_mv_user = user; g_larry_mv_mb = min_mb;
}
static ggml_sycl_larry_oproj_hook_t g_larry_oproj_hook = nullptr;
static void * g_larry_oproj_user = nullptr;
void ggml_backend_sycl_set_oproj_hook(ggml_sycl_larry_oproj_hook_t fn, void * user) {
    g_larry_oproj_hook = fn; g_larry_oproj_user = user;
}
""")
rep("sc", """static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
""", """// larry: GGML_OP_DSV41_OPROJ - the attention output's two products at decode widths: the external launcher (the xe
// tier's four-card split) when one is set and takes it, else this card alone with the kernels the two mul_mats ran
// (quant_x + fastmv Q6K_R8 for wo_a's slices and for wo_b: the same numbers)
static bool larry_oproj(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * o = dst->src[0];
    const ggml_tensor * wa = dst->src[1];
    const ggml_tensor * wb = dst->src[2];
    const int64_t OGD = wa->ne[0], OL = wa->ne[1], OG = wa->ne[2], D = wb->ne[1], KA = OL * OG;
    const int64_t n = dst->ne[1];
    if (n < 1 || n > 8 || wa->type != GGML_TYPE_Q6_K || wb->type != GGML_TYPE_Q6_K || OGD % QK_K || KA % QK_K) return false;
    queue_ptr st = ctx.stream();
    if (!g_larry_slice_planar.count(wa->data)) {   // wo_a per slice (STRATA_PREREORDER_SLICES did it at load), once
        const ggml_tensor * par = wa->view_src ? wa->view_src : wa;
        if (!reorder_qw_q6_k_moe((uint8_t *) par->data, wa->nb[2], OG, st)) return false;
        g_larry_slice_planar.insert(wa->data);
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(wb->extra);
    if (!extra) return false;
    if (!extra->optimized_feature.reorder) {   // wo_b whole (STRATA_PREREORDER did it at load), once
        if (!reorder_qw(wb, st)) return false;
        extra->optimized_feature.reorder = true;
    }
    const float * x = (const float *) o->data;
    float * y = (float *) dst->data;
    const int64_t y_cs = (int64_t) (dst->nb[1] / 4);
    if (g_larry_oproj_hook &&
        g_larry_oproj_hook(st, wa->data, wa->nb[2], wb->data, OGD, OL, OG, D, (int) n, x, OGD * OG, y, y_cs, g_larry_oproj_user))
        return true;
    ggml_sycl_pool_alloc<int8_t> xq(ctx.pool(), (size_t) (OG * n * OGD));
    ggml_sycl_pool_alloc<float> xd(ctx.pool(), (size_t) (OG * n * OGD / 32));
    ggml_sycl_pool_alloc<float> oa(ctx.pool(), (size_t) (KA * n));
    ggml_sycl_pool_alloc<int8_t> aq(ctx.pool(), (size_t) (KA * n));
    ggml_sycl_pool_alloc<float> ad(ctx.pool(), (size_t) (KA * n / 32));
    fastmv::quant_x(*st, x, OGD * OG, OGD, (int) n, OG, OGD, xq.get(), xd.get());
    fastmv::Args a;   // wo_a: slice g of o's column c -> oa[c * KA + g * OL + r] (wo_b's input column)
    a.w = wa->data; a.K = OGD; a.N = OL; a.G = OG; a.g_div = 1;
    a.n_blocks_total = OL * (OGD / QK_K); a.w_batch_bytes = wa->nb[2];
    a.x = x; a.x_col_stride = OGD * OG; a.x_batch_stride = OGD;
    a.y = oa.get(); a.y_col_stride = KA; a.y_batch_stride = OL;
    a.ncols = (int) n; a.xq = xq.get(); a.xd = xd.get();
    if (!fastmv::launch(*st, fastmv::WType::Q6K_R8, a)) return false;
    fastmv::quant_x(*st, oa.get(), KA, KA, (int) n, 1, KA, aq.get(), ad.get());
    fastmv::Args b;
    b.w = wb->data; b.K = KA; b.N = D; b.G = 1; b.g_div = 1;
    b.n_blocks_total = D * (KA / QK_K); b.w_batch_bytes = 0;
    b.x = oa.get(); b.x_col_stride = KA; b.x_batch_stride = KA;
    b.y = y; b.y_col_stride = y_cs; b.y_batch_stride = y_cs * n;
    b.ncols = (int) n; b.xq = aq.get(); b.xd = ad.get();
    return fastmv::launch(*st, fastmv::WType::Q6K_R8, b);
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
""")
rep("sc", """        case GGML_OP_DSV41_HC_FB:
            ggml_sycl_op_dsv41_hc_fb(ctx, dst);
            break;
""", """        case GGML_OP_DSV41_HC_FB:
            ggml_sycl_op_dsv41_hc_fb(ctx, dst);
            break;
        case GGML_OP_DSV41_OPROJ:
            if (!larry_oproj(ctx, dst)) GGML_ABORT("DSV41_OPROJ: shape or layout not covered");
            break;
""")
rep("sc", """        case GGML_OP_DSV41_HC_FA:
        case GGML_OP_DSV41_HC_FB:
            return true;
""", """        case GGML_OP_DSV41_HC_FA:
        case GGML_OP_DSV41_HC_FB:
            return true;
        case GGML_OP_DSV41_OPROJ:
            return op->src[1]->type == GGML_TYPE_Q6_K && op->src[2]->type == GGML_TYPE_Q6_K && op->ne[1] <= 8;
""")

# ---------------------------------------------------------------- ds41_dense.cpp
rep("dense", """static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool with_shexp = true, int64_t max_nodes = 8192) {""",
    """static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0,
                                bool with_shexp = true, int64_t max_nodes = 8192, bool dec = false) {""")
rep("dense", """    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);              // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
""", """    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * attn = nullptr;
    // larry, STRATA_TP_OPROJ (decode / verify graphs): wo_a and wo_b as one op that ggml-sycl hands to the xe tier, which
    // splits wo_a's head groups (and wo_b's matching columns) over the four cards and adds the partial sums on this one
    static const bool tp_oproj = env_on("STRATA_TP_OPROJ", false);
    if (dec && tp_oproj && im.sycl && n <= 8) {
        attn = ggml_dsv41_oproj(gc, o, wo_a, im.BL(il, "attn_output_b.weight"));                  // [D, n]
    } else {
        // grouped output LoRA (model.py: einsum over groups) then wo_b
        o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);           // [OGD, n, OG]
        ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);          // [OL, OG, n]
        attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
    }
""")
rep("dense", """        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048)).first;""",
    """        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n, 0, !split_shexp, 2048, true)).first;""")
rep("dense", """        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n);
    }
    // async (kept graphs)""", """        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n, 0, true, 8192, true);
    }
    // async (kept graphs)""")

# ---------------------------------------------------------------- the tier
rep("th", """    static bool mv_hook(void* queue, const void* w, int64_t K, int64_t N, int ncols, const int8_t* xq, const float* xd,
                        float* y, int64_t y_col_stride, void* user);
""", """    static bool mv_hook(void* queue, const void* w, int64_t K, int64_t N, int ncols, const int8_t* xq, const float* xd,
                        float* y, int64_t y_col_stride, void* user);
    /// larry (ggml-sycl's DSV41_OPROJ hook, STRATA_TP_OPROJ): the attention output split by head groups - expert card k
    /// keeps wo_a's slices of its groups and wo_b's matching columns, gets its groups' int8 o by a card-to-card copy, sends
    /// its partial sum back; the dense card computes the first groups and adds the parts.  `user` is the tier.
    static bool oproj_hook(void* queue, const void* wo_a, size_t wo_a_slice_bytes, const void* wo_b, int64_t OGD,
                           int64_t OL, int64_t OG, int64_t D, int ncols, const float* o, int64_t o_col_stride, float* dst,
                           int64_t dst_col_stride, void* user);
""")
rep("tc", """    std::map<const void*, HeadSplit> heads;
""", """    std::map<const void*, HeadSplit> heads;
    // STRATA_TP_OPROJ: per layer (keyed by wo_a), the attention output's parts - part 0 the dense card, part i >= 1 an
    // expert card: groups [g0[i], g0[i + 1]) of wo_a (its slices as they are) and wo_b's matching columns (gathered)
    struct OSplit {
        bool ok = false;
        int np = 0;
        int64_t g0[9] = {};
        std::vector<XeCard*> c;           ///< part i >= 1: c[i - 1]
        std::vector<uint8_t*> wa, wb;     ///< part i's weights (part 0's wa: wo_a itself)
        std::vector<int8_t*> xq, aq;      ///< part i >= 1: its groups' int8 o, its int8 wo_a output (8 columns)
        std::vector<float*> xd, oa, ad, y;
    };
    std::map<const void*, OSplit> osplits;
    // the dense card's side, shared by the layers (in-order on its queue): o quantised (all groups), its own part's
    // wo_a output and its quantisation, the expert cards' partial sums (8 columns each)
    int8_t* os_xq = nullptr; int8_t* os_aq = nullptr;
    float* os_xd = nullptr; float* os_oa = nullptr; float* os_ad = nullptr; float* os_slots = nullptr;
    int64_t os_calls = 0, os_declined = 0;
""")
rep("tc", """void Ds41XeTier::arm(int64_t l, int nt) {
    Ds41XeImpl& im = *im_;""", """namespace {
// STRATA_TP_OPROJ: one layer's parts (lazily, at its first decode call: no card is armed while an attention graph queues)
Ds41XeImpl::OSplit osplit_init(Ds41XeImpl& im, sycl::queue& q0, const void* wo_a, size_t sb, const void* wo_b, int64_t OGD,
                               int64_t OL, int64_t OG, int64_t D) {
    Ds41XeImpl::OSplit s;
    std::vector<XeCard*> parts;
    for (auto& cp : im.cards) if (!cp->dense) parts.push_back(cp.get());
    const int np = (int) std::min<size_t>(parts.size(), 7) + 1;
    if (np < 2 || OG % np) return s;
    const int64_t gp = OG / np, KB = OL * gp, KA = OL * OG;
    if (KB % 256 || OGD % 256) return s;
    s.np = np;
    for (int i = 0; i <= np; ++i) s.g0[i] = gp * i;
    if (!im.os_xq) {
        im.os_xq = sycl::malloc_device<int8_t>((size_t) 8 * OGD * OG, q0);
        im.os_xd = sycl::malloc_device<float>((size_t) 8 * OGD * OG / 32, q0);
        im.os_oa = sycl::malloc_device<float>((size_t) 8 * KB, q0);
        im.os_aq = sycl::malloc_device<int8_t>((size_t) 8 * KB, q0);
        im.os_ad = sycl::malloc_device<float>((size_t) 8 * KB / 32, q0);
        im.os_slots = sycl::malloc_device<float>((size_t) 8 * (np - 1) * D, q0);
        if (!im.os_xq || !im.os_xd || !im.os_oa || !im.os_aq || !im.os_ad || !im.os_slots) {
            std::fprintf(stderr, "xe tier: oproj split buffers failed - not split\\n");
            return Ds41XeImpl::OSplit();
        }
    }
    // wo_b's planar planes hold block (row r, b) at r * bpr + b: part i's columns are each row's blocks [i bpp, +bpp)
    const int64_t bpr = KA / 256, nb = D * bpr, bpp = KB / 256, nbp = D * bpp;
    const size_t po[4] = {0, 128, 192, 208}, pb[4] = {128, 64, 16, 2};
    const size_t wb_bytes = (size_t) nbp * 210, wa_bytes = (size_t) gp * sb;
    auto gather = [&](int i, uint8_t* out) {   // on the dense card: part i's columns of the four planes into out
        for (int p = 0; p < 4; ++p) {
            const uint8_t* sp = (const uint8_t*) wo_b + po[p] * (size_t) nb + pb[p] * (size_t) (i * bpp);
            uint8_t* dp = out + po[p] * (size_t) nbp;
            const size_t w = pb[p] * (size_t) bpp, spitch = pb[p] * (size_t) bpr;
            q0.parallel_for(sycl::range<1>((size_t) D * w), [=](sycl::id<1> id) {
                const size_t r = id[0] / w, b = id[0] % w;
                dp[r * w + b] = sp[r * spitch + b];
            });
        }
    };
    XeCard* dc = im.dense >= 0 ? im.cards[(size_t) im.dense].get() : nullptr;
    if (!dc) return Ds41XeImpl::OSplit();
    uint8_t* wb0 = local_alloc(q0, wb_bytes);   // the dense card's columns: read by its own kernels only
    uint8_t* stage = sycl::malloc_device<uint8_t>(wb_bytes, q0);   // the expert cards' columns, one card at a time
    if (!wb0 || !stage) {
        if (wb0) local_free(q0, wb0);
        if (stage) sycl::free(stage, q0);
        std::fprintf(stderr, "xe tier: oproj split: no room on the dense card - not split\\n");
        return Ds41XeImpl::OSplit();
    }
    gather(0, wb0);
    s.wa.push_back((uint8_t*) wo_a);
    s.wb.push_back(wb0);
    s.xq.push_back(nullptr); s.aq.push_back(nullptr);
    s.xd.push_back(nullptr); s.oa.push_back(nullptr); s.ad.push_back(nullptr); s.y.push_back(nullptr);
    bool ok = true;
    for (int i = 1; i < np && ok; ++i) {
        XeCard& c = *parts[(size_t) i - 1];
        const size_t scratch = (size_t) 8 * gp * OGD * 5 / 4 + (size_t) 8 * KB * 6 + (size_t) 8 * D * 4 + ((size_t) 1 << 20);
        if (c.free_vram() > 0 && c.free_vram() < (int64_t) (wa_bytes + wb_bytes + scratch + ((size_t) 768 << 20))) {
            std::fprintf(stderr, "xe tier: no room on card %d for the oproj split (%zu MB) - not split\\n", c.idx,
                         (wa_bytes + wb_bytes) >> 20);
            ok = false;
            break;
        }
        // its weights only its kernels read (no UR peer residency); its scratch is touched by its own engines only
        uint8_t* wa = local_alloc(c.q, wa_bytes);
        uint8_t* wb = local_alloc(c.q, wb_bytes);
        int8_t* xq = sycl::malloc_device<int8_t>((size_t) 8 * gp * OGD, c.q);
        float* xd = sycl::malloc_device<float>((size_t) 8 * gp * OGD / 32, c.q);
        float* oa = sycl::malloc_device<float>((size_t) 8 * KB, c.q);
        int8_t* aq = sycl::malloc_device<int8_t>((size_t) 8 * KB, c.q);
        float* ad = sycl::malloc_device<float>((size_t) 8 * KB / 32, c.q);
        float* y = sycl::malloc_device<float>((size_t) 8 * D, c.q);
        if (!wa || !wb || !xq || !xd || !oa || !aq || !ad || !y) {
            std::fprintf(stderr, "xe tier: oproj split allocation on card %d failed - not split\\n", c.idx);
            ok = false;
            break;
        }
        c.q.memcpy(wa, (const uint8_t*) wo_a + (size_t) (i * gp) * sb, wa_bytes);   // its slices: whole plane sets
        gather(i, stage);
        q0.wait();
        c.q.memcpy(wb, stage, wb_bytes).wait();
        s.c.push_back(&c);
        s.wa.push_back(wa); s.wb.push_back(wb);
        s.xq.push_back(xq); s.aq.push_back(aq);
        s.xd.push_back(xd); s.oa.push_back(oa); s.ad.push_back(ad); s.y.push_back(y);
    }
    q0.wait();
    sycl::free(stage, q0);
    if (!ok) return Ds41XeImpl::OSplit();   // (what was allocated stays: a few layers' worth at most, then nothing splits)
    s.ok = true;
    static int told = 0;
    if (!told++) std::fprintf(stderr, "xe tier: attention output split over %d cards (%lld groups each; %.0f MB a card a layer)\\n", np,
                              (long long) gp, (double) (wa_bytes + wb_bytes) / 1e6);
    return s;
}

// a Q6K_R8 product of G slices / one matrix for ncols columns (quant_x's int8 rows: batch-major, g * ncols + c)
void os_mv(sycl::queue& q, const void* w, int64_t K, int64_t N, int64_t G, size_t batch_bytes, int n, const int8_t* xq,
           const float* xd, float* y, int64_t y_col_stride, int64_t y_batch_stride) {
    fastmv::Args a;
    a.w = w; a.K = K; a.N = N; a.G = G; a.g_div = 1; a.n_blocks_total = N * (K / 256); a.w_batch_bytes = batch_bytes;
    a.x_col_stride = K; a.x_batch_stride = K * n;
    a.y = y; a.y_col_stride = y_col_stride; a.y_batch_stride = y_batch_stride; a.ncols = n;
    a.xq = xq; a.xd = xd;
    if (!fastmv::launch(q, fastmv::WType::Q6K_R8, a)) throw std::runtime_error("oproj split: fastmv shape not covered");
}
}  // namespace

bool Ds41XeTier::oproj_hook(void* queue, const void* wo_a, size_t sb, const void* wo_b, int64_t OGD, int64_t OL, int64_t OG,
                            int64_t D, int ncols, const float* o, int64_t ocs, float* dst, int64_t dcs, void* user) {
    Ds41XeImpl& im = *static_cast<Ds41XeTier*>(user)->im_;
    if (!queue || ncols < 1 || ncols > 8 || im.cards.empty()) return false;
    sycl::queue& q0 = *static_cast<sycl::queue*>(queue);
    // a SYCL graph recording (GGML_SYCL_GRAPH_CACHE) would capture this queue's part only: the dense card computes it all
    if (q0.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) { ++im.os_declined; return false; }
    try {
        auto it = im.osplits.find(wo_a);
        if (it == im.osplits.end()) it = im.osplits.emplace(wo_a, osplit_init(im, q0, wo_a, sb, wo_b, OGD, OL, OG, D)).first;
        Ds41XeImpl::OSplit& s = it->second;
        if (!s.ok) { ++im.os_declined; return false; }
        const int n = ncols;
        const int64_t gp = s.g0[1], KB = OL * gp;
        std::lock_guard<PrioMutex> lk(g_submit);
        const sycl::event e0 = fastmv::quant_x(q0, o, ocs, OGD, n, OG, OGD, im.os_xq, im.os_xd);   // all groups
        std::vector<sycl::event> ev;
        for (int i = 1; i < s.np; ++i) {
            XeCard& c = *s.c[(size_t) i - 1];
            const size_t r0 = (size_t) s.g0[i] * n;   // its groups' rows of the quantised o
            int8_t* xq = s.xq[(size_t) i];
            const int8_t* xq_src = im.os_xq + r0 * OGD;
            c.q.submit([&](sycl::handler& h) { h.depends_on(e0); h.memcpy(xq, xq_src, (size_t) gp * n * OGD); });
            c.q.memcpy(s.xd[(size_t) i], im.os_xd + r0 * (OGD / 32), (size_t) gp * n * (OGD / 32) * sizeof(float));
            os_mv(c.q, s.wa[(size_t) i], OGD, OL, gp, sb, n, xq, s.xd[(size_t) i], s.oa[(size_t) i], KB, OL);
            fastmv::quant_x(c.q, s.oa[(size_t) i], KB, KB, n, 1, KB, s.aq[(size_t) i], s.ad[(size_t) i]);
            os_mv(c.q, s.wb[(size_t) i], KB, D, 1, 0, n, s.aq[(size_t) i], s.ad[(size_t) i], s.y[(size_t) i], D, D * n);
            ev.push_back(c.q.memcpy(im.os_slots + (size_t) (i - 1) * 8 * D, s.y[(size_t) i], (size_t) n * D * sizeof(float)));
        }
        os_mv(q0, s.wa[0], OGD, OL, gp, sb, n, im.os_xq, im.os_xd, im.os_oa, KB, OL);   // its groups: o's first rows
        fastmv::quant_x(q0, im.os_oa, KB, KB, n, 1, KB, im.os_aq, im.os_ad);
        os_mv(q0, s.wb[0], KB, D, 1, 0, n, im.os_aq, im.os_ad, dst, dcs, dcs * n);
        const float* sl = im.os_slots;
        const int nparts = s.np - 1;
        q0.submit([&](sycl::handler& h) {   // dst += the expert cards' partial sums
            h.depends_on(ev);
            h.parallel_for(sycl::range<1>((size_t) n * D), [=](sycl::id<1> id) {
                const int64_t c = (int64_t) id[0] / D, r = (int64_t) id[0] % D;
                float v = dst[c * dcs + r];
                for (int k = 0; k < nparts; ++k) v += sl[(size_t) k * 8 * D + c * D + r];
                dst[c * dcs + r] = v;
            });
        });
        ++im.os_calls;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xe tier: oproj split: %s\\n", e.what());
        return false;
    }
}

void Ds41XeTier::arm(int64_t l, int nt) {
    Ds41XeImpl& im = *im_;""")
# the counts at close
rep("tc", """    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\\n", (long long) im.arm_used,""",
    """    if (im.os_calls || im.os_declined)
        std::fprintf(stderr, "xe tier: attention output split %lld calls (%zu layers), declined %lld\\n", (long long) im.os_calls,
                     im.osplits.size(), (long long) im.os_declined);
    if (im.arm_used || im.arm_disarmed)
        std::fprintf(stderr, "xe tier: armed card shares used %lld, disarmed calls %lld\\n", (long long) im.arm_used,""")

# ---------------------------------------------------------------- ds41_generate.cpp
rep("gen", """    if (Ds41XeTier::tl_on()) ds41_dense_tl_hook = [](int id) { tl_tier->tl_tag(id); };
#else""", """    // larry, STRATA_TP_OPROJ=1: the attention output split over the cards (the dense graphs build GGML_OP_DSV41_OPROJ)
    if (!mc.cpu_only && std::getenv("STRATA_TP_OPROJ") && std::atoi(std::getenv("STRATA_TP_OPROJ")) != 0)
        ggml_backend_sycl_set_oproj_hook(&Ds41XeTier::oproj_hook, &tier);
    if (Ds41XeTier::tl_on()) ds41_dense_tl_hook = [](int id) { tl_tier->tl_tag(id); };
#else""")

for k, p in FILES.items():
    open(p, "w").write(src[k])
print("patched")
