#!/usr/bin/env python3
# patch-inq.py - larry, GGML_SYCL_LARRY_INQ (default 1): decode's (1-column) Q6K_R8 mat-vecs quantise their f32 input
# themselves - each work-group into local memory, quant_x's exact rounding (int8 per 32 values, amax / 127) - instead
# of a separate quant_x kernel per input: a decode layer ran 6 of them (~2.5-2.9 us each + a ~2.8 us dependent-launch
# gap).  Bit-identical to the quant_x path (GGML_SYCL_LARRY_INQ_CHECK=1 compares).  Idempotent.
import sys
G = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/"


def rep(path, old, new, cnt=1):
    s = open(path).read()
    if new.strip() in s:
        print(f"{path}: already"); return
    if s.count(old) != cnt: sys.exit(f"{path}: anchor x{s.count(old)}: {old[:90]!r}")
    open(path, "w").write(s.replace(old, new))
    print(f"{path}: patched")


H = G + "fastmv.hpp"
rep(H, """    const int8_t * xq = nullptr;
    const float * xd = nullptr;
};""", """    const int8_t * xq = nullptr;
    const float * xd = nullptr;
    // larry, Q6K_R8 with 1 column: the kernel quantises x itself (each work-group into local memory; xq / xd unused)
    bool inq = false;
};""")

C = G + "fastmv.cpp"
# the shared 32-value quantiser (quant_x's math)
rep(C, """// the Q6K_R unit (32 values) against int8 activations: one dp4a per 4 values
template <int NT>
inline void q6k_unit_q8(""", """// larry: quant_x's rounding of one 32-value block (int8 = rint(v * 127 / amax), scale amax / 127) - the in-kernel
// quantisation (Args::inq) writes local memory with it, so its results match the quant_x path bit for bit
inline void quant32(const float * p, int8_t * oq, float * od) {
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const sycl::float4 f = reinterpret_cast<const sycl::float4 *>(p)[i];
        v[4 * i] = f.x(); v[4 * i + 1] = f.y(); v[4 * i + 2] = f.z(); v[4 * i + 3] = f.w();
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
    const float id_ = amax > 0.f ? 127.0f / amax : 0.f;
    uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 32; ++i) w[i / 4] |= (uint32_t) ((int) sycl::rint(v[i] * id_) & 0xFF) << (8 * (i % 4));
    sycl::uint4 * o = reinterpret_cast<sycl::uint4 *>(oq);
    o[0] = sycl::uint4(w[0], w[1], w[2], w[3]);
    o[1] = sycl::uint4(w[4], w[5], w[6], w[7]);
    *od = amax / 127.0f;
}

// the Q6K_R unit (32 values) against int8 activations: one dp4a per 4 values
template <int NT>
inline void q6k_unit_q8(""")

# generic kernel: template INQ, local buffers, prologue, the Q6K_R8 branch's pointers
rep(C, """template <WType T, int NT, int LPR>
void run(sycl::queue & q, const Args & a, int KS, int RW) {""", """template <WType T, int NT, int LPR, bool INQ = false>
void run(sycl::queue & q, const Args & a, int KS, int RW) {""")
rep(C, """    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (KS > 1 ? RW * KS * NT : 1)), h);
        h.parallel_for(sycl::nd_range<2>(gr, lr), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t g = (int64_t) it.get_global_id(0);
            const int lid = (int) it.get_local_id(1);
            const int grp = lid / LPR;                 // (row in wg, k-slice)""", """    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (KS > 1 ? RW * KS * NT : 1)), h);
        // larry, INQ: this batch's columns quantised into local memory by the work-group (16-byte aligned views)
        sycl::local_accessor<sycl::uint4, 1> sxq(sycl::range<1>(INQ ? (size_t) (NT * K / 16) : 1), h);
        sycl::local_accessor<sycl::float4, 1> sxd(sycl::range<1>(INQ ? (size_t) (NT * K / 128) : 1), h);
        h.parallel_for(sycl::nd_range<2>(gr, lr), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t g = (int64_t) it.get_global_id(0);
            const int lid = (int) it.get_local_id(1);
            int8_t * sq = nullptr;
            float * sd = nullptr;
            if constexpr (INQ) {
                sq = reinterpret_cast<int8_t *>(&sxq[0]);
                sd = reinterpret_cast<float *>(&sxd[0]);
                const int nj = (int) (K / 32), nblk = A.ncols * nj;
                for (int bi = lid; bi < nblk; bi += lanes_per_wg) {
                    const int t = bi / nj, j = bi % nj;
                    quant32(A.x + g * A.x_batch_stride + (int64_t) t * A.x_col_stride + 32 * j, sq + (int64_t) t * K + 32 * j,
                            sd + (int64_t) t * nj + j);
                }
                sycl::group_barrier(it.get_group());
            }
            const int grp = lid / LPR;                 // (row in wg, k-slice)""")
rep(C, """                    for (int t = 0; t < NT; ++t) {
                        const int64_t c = g * A.ncols + (t < A.ncols ? t : 0);   // quant_x's rows: batch-major columns
                        xq[t] = A.xq + c * K;
                        xd[t] = A.xd + c * (K / 32);
                    }""", """                    for (int t = 0; t < NT; ++t) {
                        const int64_t c = g * A.ncols + (t < A.ncols ? t : 0);   // quant_x's rows: batch-major columns
                        if constexpr (INQ) {
                            const int64_t tc = t < A.ncols ? t : 0;
                            xq[t] = sq + tc * K;
                            xd[t] = sd + tc * (K / 32);
                        } else {
                            xq[t] = A.xq + c * K;
                            xd[t] = A.xd + c * (K / 32);
                        }
                    }""")

# row-blocked kernel
rep(C, """template <int NT, int LPR, int R, bool BIG>   // BIG: the 256-register mode (half the threads per EU, no spills)
void run_r8m(sycl::queue & q, const Args & a, int KS, int RW) {""", """template <int NT, int LPR, int R, bool BIG, bool INQ = false>   // BIG: the 256-register mode (half the threads per EU, no spills)
void run_r8m(sycl::queue & q, const Args & a, int KS, int RW) {""")
rep(C, """    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (KS > 1 ? RW * KS * R * NT : 1)), h);
        auto kern = [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t g = (int64_t) it.get_global_id(0);
            const int lid = (int) it.get_local_id(1);""", """    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) (KS > 1 ? RW * KS * R * NT : 1)), h);
        sycl::local_accessor<sycl::uint4, 1> sxq(sycl::range<1>(INQ ? (size_t) (NT * K / 16) : 1), h);   // larry, INQ
        sycl::local_accessor<sycl::float4, 1> sxd(sycl::range<1>(INQ ? (size_t) (NT * K / 128) : 1), h);
        auto kern = [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t g = (int64_t) it.get_global_id(0);
            const int lid = (int) it.get_local_id(1);
            int8_t * sq = nullptr;
            float * sd = nullptr;
            if constexpr (INQ) {
                sq = reinterpret_cast<int8_t *>(&sxq[0]);
                sd = reinterpret_cast<float *>(&sxd[0]);
                const int nj = (int) (K / 32), nblk = A.ncols * nj;
                for (int bi = lid; bi < nblk; bi += lanes_per_wg) {
                    const int t = bi / nj, j = bi % nj;
                    quant32(A.x + g * A.x_batch_stride + (int64_t) t * A.x_col_stride + 32 * j, sq + (int64_t) t * K + 32 * j,
                            sd + (int64_t) t * nj + j);
                }
                sycl::group_barrier(it.get_group());
            }""")
rep(C, """            for (int t = 0; t < NT; ++t) {
                const int64_t c = g * A.ncols + t;   // NT == ncols: exact
                xq[t] = A.xq + c * K;
                xd[t] = A.xd + c * (K / 32);
            }""", """            for (int t = 0; t < NT; ++t) {
                const int64_t c = g * A.ncols + t;   // NT == ncols: exact
                if constexpr (INQ) {
                    xq[t] = sq + (int64_t) t * K;
                    xd[t] = sd + (int64_t) t * (K / 32);
                } else {
                    xq[t] = A.xq + c * K;
                    xd[t] = A.xd + c * (K / 32);
                }
            }""")
# dispatch: pick_r8m and pick pass INQ for the Q6K_R8 one-column case
rep(C, """    if (lpr == 16) run_r8m<NT, 16, R, BIG>(q, a, ks, rw);
    else if (lpr == 8) run_r8m<NT, 8, R, BIG>(q, a, ks, rw);
    else run_r8m<NT, 4, R, BIG>(q, a, ks, rw);""", """    if constexpr (NT == 1) {
        if (a.inq) {   // larry: the in-kernel quantisation (one column)
            if (lpr == 16) run_r8m<NT, 16, R, BIG, true>(q, a, ks, rw);
            else if (lpr == 8) run_r8m<NT, 8, R, BIG, true>(q, a, ks, rw);
            else run_r8m<NT, 4, R, BIG, true>(q, a, ks, rw);
            return;
        }
    }
    if (lpr == 16) run_r8m<NT, 16, R, BIG>(q, a, ks, rw);
    else if (lpr == 8) run_r8m<NT, 8, R, BIG>(q, a, ks, rw);
    else run_r8m<NT, 4, R, BIG>(q, a, ks, rw);""")
rep(C, """    while ((rw * ks * lpr) % SG) ++rw;
    if (lpr == 16) run<T, NT, 16>(q, a, ks, rw);
    else if (lpr == 8) run<T, NT, 8>(q, a, ks, rw);
    else run<T, NT, 4>(q, a, ks, rw);""", """    while ((rw * ks * lpr) % SG) ++rw;
    if constexpr (T == WType::Q6K_R8 && NT == 1) {
        if (a.inq) {   // larry: the in-kernel quantisation (one column)
            if (lpr == 16) run<T, NT, 16, true>(q, a, ks, rw);
            else if (lpr == 8) run<T, NT, 8, true>(q, a, ks, rw);
            else run<T, NT, 4, true>(q, a, ks, rw);
            return;
        }
    }
    if (lpr == 16) run<T, NT, 16>(q, a, ks, rw);
    else if (lpr == 8) run<T, NT, 8>(q, a, ks, rw);
    else run<T, NT, 4>(q, a, ks, rw);""")
rep(C, """    if (t == WType::Q6K_R8 && (!a.xq || !a.xd)) return false;""",
    """    if (t == WType::Q6K_R8 && a.inq && a.ncols != 1) return false;   // larry: in-kernel quantisation: one column
    if (t == WType::Q6K_R8 && !a.inq && (!a.xq || !a.xd)) return false;""")

S = G + "ggml-sycl.cpp"
rep(S, """    if (t == fastmv::WType::Q6K_R8 && fastmv::xmx_wants(a)) {""", """    // larry, GGML_SYCL_LARRY_INQ (default 1): one column - the mat-vec quantises its input itself (no quant_x kernel)
    static const int inq = ggml_sycl_get_env("GGML_SYCL_LARRY_INQ", 1);
    if (t == fastmv::WType::Q6K_R8 && inq && ncols == 1 &&
        !(g_larry_mv_hook && G == 1 && src0->ne[2] == 1 && (double) N * K * 210.0 / 256.0 >= g_larry_mv_mb * 1e6)) {
        static const int inq_chk = ggml_sycl_get_env("GGML_SYCL_LARRY_INQ_CHECK", 0);   // diagnostic: vs quant_x
        if (inq_chk) {
            const size_t ny = (size_t) (N * G);
            ggml_sycl_pool_alloc<float> ty(ctx.pool(), ny);
            fastmv::Args b = a;
            b.inq = true; b.xq = nullptr; b.xd = nullptr;
            b.y = ty.get(); b.y_col_stride = N; b.y_batch_stride = N;
            if (!fastmv::launch(*ctx.stream(), t, b)) GGML_ABORT("inq check: launch");
            ggml_sycl_pool_alloc<int8_t> rq(ctx.pool(), (size_t) (G * K));
            ggml_sycl_pool_alloc<float> rd(ctx.pool(), (size_t) (G * K / 32));
            fastmv::quant_x(*ctx.stream(), a.x, a.x_col_stride, a.x_batch_stride, 1, G, K, rq.get(), rd.get());
            fastmv::Args r = a;
            r.xq = rq.get(); r.xd = rd.get();
            fastmv::launch(*ctx.stream(), t, r);
            static int nchk = 0;
            static long nbad = 0, ncall = 0;
            std::vector<float> va(ny), vb(ny);
            ctx.stream()->wait();
            for (int64_t g2 = 0; g2 < G; ++g2) ctx.stream()->memcpy(va.data() + g2 * N, a.y + g2 * a.y_batch_stride, N * 4);
            ctx.stream()->memcpy(vb.data(), ty.get(), ny * 4).wait();
            size_t nd = 0;
            for (size_t i = 0; i < ny; ++i) nd += va[i] != vb[i];
            ++ncall;
            if (nd) ++nbad;
            if (nchk < 12 || (nd && nbad <= 12)) {
                ++nchk;
                GGML_LOG_INFO("larry inq check: %s %lld x %lld (G %lld): %zu of %zu outputs differ\\n", dst->name, (long long) K,
                              (long long) N, (long long) G, nd, ny);
            }
            if (ncall % 2000 == 0) GGML_LOG_INFO("larry inq check: %ld calls, %ld with differences\\n", ncall, nbad);
            return true;
        }
        a.inq = true;
        a.xq = nullptr;
        a.xd = nullptr;
        return fastmv::launch(*ctx.stream(), t, a);
    }
    if (t == fastmv::WType::Q6K_R8 && fastmv::xmx_wants(a)) {""")
