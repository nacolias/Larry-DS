#!/usr/bin/env python3
# patch-woa.py - larry: wo_a (a 2-D Q6_K weight read through 8 slice views) reordered per slice at load
# (ggml_backend_sycl_prereorder_slices), so every prompt span takes larry_slices_gemm: before the first decode token,
# ggml-sycl's generic path gathered each group's permuted activations with a 2-D dpct copy at ~45 GB/s (8 x 188 us a
# layer-span: ~11 % of a long one-shot prompt's card-0 time).  larry_slices_gemm's per-slice f16 dequant goes through the
# fp16 weight cache (a 16K batch's 32 spans read each layer's wo_a).  Idempotent.
import sys
G = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/"
S = G + "src/ggml-sycl/ggml-sycl.cpp"
EN = "/opt/ai/dsv41/xolotl/tools/v41/engine/"


def rep(path, old, new, cnt=1):
    s = open(path).read()
    if new.strip() in s:
        print(f"{path}: already"); return
    if s.count(old) != cnt: sys.exit(f"{path}: anchor x{s.count(old)}: {old[:90]!r}")
    open(path, "w").write(s.replace(old, new))
    print(f"{path}: patched")


rep(G + "include/ggml-sycl.h", """GGML_BACKEND_API bool ggml_backend_sycl_prereorder(ggml_backend_t backend, struct ggml_tensor * t);
""", """GGML_BACKEND_API bool ggml_backend_sycl_prereorder(ggml_backend_t backend, struct ggml_tensor * t);
// larry: a 2-D Q6_K weight read only through n_slices 3-D slice views (wo_a) reordered per slice now - the planar layout
// decode's mat-vec and the prompt's per-slice GEMM read - instead of by the first decode-shaped product
GGML_BACKEND_API bool ggml_backend_sycl_prereorder_slices(ggml_backend_t backend, struct ggml_tensor * t, int64_t n_slices);
""")

# the cache: a core any conversion can use, the leaf check in the op_mul_mat wrapper
rep(S, """// the fp16 copy of weight src (ne values), converted now if it is not cached; nullptr = not cacheable / over budget
static const sycl::half * larry_f16cache_get(const ggml_tensor * src0, const void * src, size_t ne, to_fp16_sycl_t conv,
                                             queue_ptr stream) {
    static const double mb = [] { const char * e = std::getenv("GGML_SYCL_LARRY_F16CACHE_MB"); return e ? std::atof(e) : 768.0; }();
    const size_t budget = (size_t) (mb * 1048576.0);
    if (budget == 0 || src0->op != GGML_OP_NONE || src0->view_src != nullptr || ne * 2 > budget) return nullptr;
""", """// the fp16 copy of the weight data at src (ne values), conv(y) writing it if it is not cached; nullptr = over budget
template <typename F>
static const sycl::half * larry_f16cache_core(const void * src, size_t ne, F && conv, queue_ptr stream) {
    static const double mb = [] { const char * e = std::getenv("GGML_SYCL_LARRY_F16CACHE_MB"); return e ? std::atof(e) : 768.0; }();
    const size_t budget = (size_t) (mb * 1048576.0);
    if (budget == 0 || ne * 2 > budget) return nullptr;
""")
rep(S, """    conv(src, slot->f16, (int64_t) ne, stream);
    slot->src = src;
    slot->ne = ne;
    slot->tick = ++g_larry_f16c_tick;
    return slot->f16;
}
""", """    conv(slot->f16);
    slot->src = src;
    slot->ne = ne;
    slot->tick = ++g_larry_f16c_tick;
    return slot->f16;
}

// the fp16 copy of weight src0 (a leaf, not a view; its data at src, ne values); nullptr = not cacheable / over budget
static const sycl::half * larry_f16cache_get(const ggml_tensor * src0, const void * src, size_t ne, to_fp16_sycl_t conv,
                                             queue_ptr stream) {
    if (src0->op != GGML_OP_NONE || src0->view_src != nullptr) return nullptr;
    return larry_f16cache_core(src, ne, [&](sycl::half * y) { conv(src, y, (int64_t) ne, stream); }, stream);
}
""")

# larry_slices_gemm: the slices' f16 copies through the cache
rep(S, """    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> wf(ctx.pool(), (size_t) N * K);
    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);
    const int64_t g_div = G / src0->ne[2];
    for (int64_t g = 0; g < G; ++g) {
        if (g % g_div == 0)
            fastmv::dequant_q6k_planar_f16(*st, (const char *) src0->data + (g / g_div) * src0->nb[2], N, K, wf.get());
""", """    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> wf(ctx.pool());   // only when the fp16 weight cache cannot hold the slice
    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);
    const int64_t g_div = G / src0->ne[2];
    const sycl::half * w = nullptr;
    for (int64_t g = 0; g < G; ++g) {
        if (g % g_div == 0) {
            const char * sl = (const char *) src0->data + (g / g_div) * src0->nb[2];
            w = larry_f16cache_core(sl, (size_t) (N * K), [&](sycl::half * y) { fastmv::dequant_q6k_planar_f16(*st, sl, N, K, y); }, st);
            if (!w) {
                if (!wf.get()) wf.alloc((size_t) N * K);
                fastmv::dequant_q6k_planar_f16(*st, sl, N, K, wf.get());
                w = wf.get();
            }
        }
""")
rep(S, """                                           xf.get(), K, wf.get(), K, 0.0f,
                                           (float *) ((char *) dst->data + g * dst->nb[2]), (int64_t) (dst->nb[1] / 4));""",
    """                                           xf.get(), K, w, K, 0.0f,
                                           (float *) ((char *) dst->data + g * dst->nb[2]), (int64_t) (dst->nb[1] / 4));""")

# the load-time per-slice reorder
rep(S, """// prompt chunks against a per-slice planar Q6_K weight: per slice, W (f16) and the slice's columns (f16), one GEMM
""", """bool ggml_backend_sycl_prereorder_slices(ggml_backend_t backend, ggml_tensor * t, int64_t n_slices) {
    // the knobs larry_fastmv reads a planar slice through: without them a reorder here would leave it unreadable
    static const int on = ggml_sycl_get_env("GGML_SYCL_LARRY_FASTMV", 1);
    static const int q6r8 = ggml_sycl_get_env("GGML_SYCL_LARRY_Q6R8", 1);
    static const int woa = ggml_sycl_get_env("GGML_SYCL_LARRY_WOA", 1);
    if (!on || !q6r8 || !woa || !backend || !t || t->type != GGML_TYPE_Q6_K || t->view_src || t->ne[2] != 1 ||
        t->ne[3] != 1 || n_slices < 2 || t->ne[1] % n_slices || t->ne[0] % QK_K || !t->extra) return false;
    if (g_larry_slice_planar.count(t->data)) return true;
    if (static_cast<const ggml_tensor_extra_gpu *>(t->extra)->optimized_feature.reorder) return false;   // reordered whole
    auto * ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    if (!reorder_qw_q6_k_moe((uint8_t *) t->data, ggml_nbytes(t) / n_slices, n_slices, ctx->stream())) return false;
    g_larry_slice_planar.insert(t->data);
    return true;
}

// prompt chunks against a per-slice planar Q6_K weight: per slice, W (f16) and the slice's columns (f16), one GEMM
""")

# the engine: the dense model lists them, the generator reorders them at load
rep(EN + "ds41_dense.hpp", """    std::vector<ggml_tensor *> reorder_weights() const;
""", """    std::vector<ggml_tensor *> reorder_weights() const;
    /// larry: the 2-D Q6_K weights read only through per-group slice views (wo_a) - for ggml_backend_sycl_prereorder_slices
    std::vector<ggml_tensor *> slice_reorder_weights() const;
""")
rep(EN + "ds41_dense.cpp", """// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}""", """// ================================================================== public API

std::vector<ggml_tensor *> Ds41Dense::slice_reorder_weights() const {
    std::vector<ggml_tensor *> v;
    const std::string suf = "attn_output_a.weight";
    for (const auto & kv : p_->w.t) {
        ggml_tensor * t = kv.second;
        if (t->type != GGML_TYPE_Q6_K || t->ne[2] != 1 || t->ne[3] != 1 || !t->buffer) continue;
        if (kv.first.size() > suf.size() && kv.first.compare(kv.first.size() - suf.size(), suf.size(), suf) == 0) v.push_back(t);
    }
    return v;
}

Ds41Dense::Ds41Dense() : p_(new Impl()) {}""")
rep(EN + "ds41_generate.cpp", """        for (ggml_tensor * t : dense.reorder_weights()) nr += ggml_backend_sycl_prereorder(dense.backend(), t) ? 1 : 0;
        ggml_backend_synchronize(dense.backend());
        std::fprintf(stderr, "dense: %d Q6_K weights reordered at load (%.2f s)\\n", nr, (now_ms() - t0) / 1000.0);""",
    """        for (ggml_tensor * t : dense.reorder_weights()) nr += ggml_backend_sycl_prereorder(dense.backend(), t) ? 1 : 0;
        // wo_a per group slice: every prompt span then takes the per-slice GEMM, from the first prompt on
        int ns = 0;
        for (ggml_tensor * t : dense.slice_reorder_weights())
            ns += ggml_backend_sycl_prereorder_slices(dense.backend(), t, dense.geom().o_groups) ? 1 : 0;
        ggml_backend_synchronize(dense.backend());
        std::fprintf(stderr, "dense: %d Q6_K weights reordered at load, %d per slice (%.2f s)\\n", nr, ns, (now_ms() - t0) / 1000.0);""")
