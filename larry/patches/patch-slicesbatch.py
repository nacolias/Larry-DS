#!/usr/bin/env python3
# patch-slicesbatch.py [GGML_SYCL_CPP] - larry, GGML_SYCL_LARRY_SLICES_BATCH (default 0): larry_slices_gemm (a prompt
# chunk against a per-slice planar Q6_K weight: wo_a, 8 slices of 4096 -> 1024) as ONE strided gemm_batch.  It ran each
# slice on its own - its columns gathered to fp16, one oneMKL GEMM - 8 conversions, 8 GEMMs and their submissions a
# call (768 us at 512 columns, 277 of host time: 8.4 % of card 0 in a 31K prompt's second batch, abwT118).  When src1 is
# the permuted view of a contiguous [ncols][G][K] tensor (the prompt path's o), it is converted once in place order,
# the parent's slices dequantised into one fp16 cache entry, and one gemm_batch writes every slice of dst (the same
# layout, the same fp16 x fp16 -> f32 products).  Otherwise the per-slice loop as before.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(P).read()
if "GGML_SYCL_LARRY_SLICES_BATCH" in s:
    print("already patched"); sys.exit(0)
old = """    if (src1->nb[0] != 4 || dst->nb[0] != 4) return false;
    dpct::queue_ptr st = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);"""
new = """    if (src1->nb[0] != 4 || dst->nb[0] != 4) return false;
    dpct::queue_ptr st = ctx.stream();
    // larry, patch-slicesbatch.py: every slice in one gemm_batch when src1's memory is a contiguous [ncols][G][K]
    static const int batched = ggml_sycl_get_env("GGML_SYCL_LARRY_SLICES_BATCH", 0);
    if (batched && G == src0->ne[2] && src1->nb[2] == (size_t) K * 4 && src1->nb[1] == (size_t) (G * K) * 4 &&
        dst->nb[1] % 4 == 0 && dst->nb[2] % 4 == 0) {
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
    }
    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
