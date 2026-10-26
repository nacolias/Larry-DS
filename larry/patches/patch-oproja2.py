#!/usr/bin/env python3
# patch-oproja2.py [GGML_SYCL_CPP] - larry: larry_slices_batched (patch-oproja.py) when the slices' outputs interleave
# (wo_b's layout: slice g's [ncols][N] at g * N with ld N * G): oneMKL's strided gemm_batch rejects that stride
# ("Illegal value supplied for parameter stride_c": abwT121's STRATA_OPROJ_A runs aborted), so the pointer form - one
# group of G products, the device pointer arrays written by a kernel - and, should that throw too, one gemm a slice.
# Needs patch-oproja.py first.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "patch-oproja2.py" in s:
    print("already patched"); sys.exit(0)
old = """    oneapi::mkl::blas::row_major::gemm_batch(*st, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, ncols, N, K,
                                             1.0f, xh, G * K, K, w, K, N * K, 0.0f, dd, ldc, sc, G);
}"""
new = """    if (sc >= ldc * ncols) {   // separate slices: the strided batch
        oneapi::mkl::blas::row_major::gemm_batch(*st, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, ncols, N, K,
                                                 1.0f, xh, G * K, K, w, K, N * K, 0.0f, dd, ldc, sc, G);
        return;
    }
    // larry, patch-oproja2.py: interleaved slices (wo_b's layout) - oneMKL's strided batch rejects a stride below ldc *
    // rows, so the pointer form: one group of G products, the device pointer arrays written by a kernel
    ggml_sycl_pool_alloc<const sycl::half *> pa(ctx.pool(), (size_t) G), pb(ctx.pool(), (size_t) G);
    ggml_sycl_pool_alloc<float *> pc(ctx.pool(), (size_t) G);
    const sycl::half ** ap = pa.get();
    const sycl::half ** bp = pb.get();
    float ** cp = pc.get();
    st->parallel_for(sycl::range<1>((size_t) G), [=](sycl::id<1> i) {
        const int64_t g = (int64_t) i[0];
        ap[g] = xh + g * K;
        bp[g] = w + g * N * K;
        cp[g] = dd + g * sc;
    });
    oneapi::mkl::transpose ta = oneapi::mkl::transpose::nontrans, tb = oneapi::mkl::transpose::trans;
    int64_t m_ = ncols, n_ = N, k_ = K, la = G * K, lb = K, lc = ldc, gs = G;
    float al = 1.0f, be = 0.0f;
    try {
        oneapi::mkl::blas::row_major::gemm_batch(*st, &ta, &tb, &m_, &n_, &k_, &al, ap, &la, bp, &lb, &be, cp, &lc, (int64_t) 1, &gs);
    } catch (const std::exception & e) {   // one gemm a slice
        static bool said = false;
        if (!said) { said = true; GGML_LOG_WARN("larry_slices_batched: pointer gemm_batch failed (%s) - a gemm a slice\\n", e.what()); }
        for (int64_t g = 0; g < G; ++g)
            oneapi::mkl::blas::row_major::gemm(*st, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, ncols, N, K, 1.0f,
                                               xh + g * K, G * K, w + g * N * K, K, 0.0f, dd + g * sc, ldc);
    }
}"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
