#!/usr/bin/env python3
# patch-argmax.py - larry, GGML_SYCL_LARRY_ARGMAX (default 1): argmax of a few LONG rows (the DSpark draft's Markov
# head: 129280 logits, 5 a draft) over many work-groups - ggml-sycl's kernel runs one 256-lane work-group a row,
# ~500 dependent compares a lane: ~60 us a row.  Two kernels: per-chunk (value, index) partials, then one work-group a
# row; ties keep the smaller index (the first maximum, as the CPU op).  Idempotent.
import sys
S = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(S).read()
if "GGML_SYCL_LARRY_ARGMAX" in s: print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {""", r'''// larry, GGML_SYCL_LARRY_ARGMAX: rows of >= 16384 values, <= 64 rows - NC chunks a row (256 lanes each), then a
// reduction of the chunks; (v1, i1) beats (v2, i2) iff v1 > v2 or (v1 == v2 and i1 < i2)
static bool larry_argmax(ggml_backend_sycl_context & ctx, const float * x, int32_t * dst, int64_t ncols, int64_t nrows,
                         dpct::queue_ptr st) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_LARRY_ARGMAX", 1);
    if (!on || ncols < 16384 || nrows < 1 || nrows > 64 || ncols > INT32_MAX) return false;
    const int NC = (int) std::min<int64_t>(64, ncols / 2048);
    ggml_sycl_pool_alloc<float> pv(ctx.pool(), (size_t) (nrows * NC));
    ggml_sycl_pool_alloc<int32_t> pi(ctx.pool(), (size_t) (nrows * NC));
    float * PV = pv.get();
    int32_t * PI = pi.get();
    const int64_t per = (ncols + NC - 1) / NC;
    auto better = [](float v1, int32_t i1, float v2, int32_t i2) { return v1 > v2 || (v1 == v2 && i1 >= 0 && (i2 < 0 || i1 < i2)); };
    st->submit([&](sycl::handler & h) {
    sycl::local_accessor<float, 1> lv(sycl::range<1>(16), h);
    sycl::local_accessor<int32_t, 1> li(sycl::range<1>(16), h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) nrows, (size_t) NC * 256), sycl::range<2>(1, 256)),
                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t r = it.get_global_id(0);
        const int c = (int) it.get_group(1), tid = (int) it.get_local_id(1);
        const int64_t c0 = c * per, c1 = sycl::min(ncols, c0 + per);
        float bv = -INFINITY;
        int32_t bi = -1;
        for (int64_t k = c0 + tid; k < c1; k += 256) {   // ascending: strict > keeps this lane's first maximum
            const float v = x[r * ncols + k];
            if (v > bv || bi < 0) { bv = v; bi = (int32_t) k; }
        }
        const sycl::sub_group sg = it.get_sub_group();
        for (int off = 8; off > 0; off >>= 1) {
            const float ov = sycl::permute_group_by_xor(sg, bv, off);
            const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        // 16 sub-groups: lane 0 of each writes, the first sub-group reduces them
        const int sgi = (int) sg.get_group_linear_id();
        if (sg.get_local_linear_id() == 0) { lv[sgi] = bv; li[sgi] = bi; }
        it.barrier(sycl::access::fence_space::local_space);
        if (sgi == 0) {
            const int l = (int) sg.get_local_linear_id();
            bv = lv[l]; bi = li[l];
            for (int off = 8; off > 0; off >>= 1) {
                const float ov = sycl::permute_group_by_xor(sg, bv, off);
                const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
                if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (l == 0) { PV[r * NC + c] = bv; PI[r * NC + c] = bi; }
        }
    });
    });
    st->submit([&](sycl::handler & h) {
    sycl::local_accessor<float, 1> lv(sycl::range<1>(4), h);
    sycl::local_accessor<int32_t, 1> li(sycl::range<1>(4), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * 64), sycl::range<1>(64)),
                     [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        float bv = tid < NC ? PV[r * NC + tid] : -INFINITY;
        int32_t bi = tid < NC ? PI[r * NC + tid] : -1;
        const sycl::sub_group sg = it.get_sub_group();
        for (int off = 8; off > 0; off >>= 1) {
            const float ov = sycl::permute_group_by_xor(sg, bv, off);
            const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        const int sgi = (int) sg.get_group_linear_id();
        if (sg.get_local_linear_id() == 0) { lv[sgi] = bv; li[sgi] = bi; }
        it.barrier(sycl::access::fence_space::local_space);
        if (tid == 0) {
            for (int k = 1; k < 4; ++k)
                if (better(lv[k], li[k], bv, bi)) { bv = lv[k]; bi = li[k]; }
            // lv[0] / li[0] = this sub-group's own result (bv, bi)
            dst[r] = bi;
        }
    });
    });
    return true;
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {''')
rep("""    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}""", """    if (larry_argmax(ctx, src0_dd, dst_dd, ncols, nrows, main_stream)) return;
    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}""")
open(S, "w").write(s)
print("patched")
