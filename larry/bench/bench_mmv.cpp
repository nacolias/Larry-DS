// bench_mmv.cpp - ggml-sycl's mat-vec speed on one B70 at DeepSeek-V4.1's dense shapes, per weight type.
//   bench_mmv [ncols=1] [iters=200]
// One graph = ONE mul_mat (weights [K x N] of the type, x [K x ncols] F32); reports us per call and GB/s of weights.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int ncols = argc > 1 ? atoi(argv[1]) : 1;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    ggml_backend_t be = ggml_backend_sycl_init(0);
    if (!be) { std::fprintf(stderr, "no sycl backend\n"); return 1; }
    struct Shape { const char* name; int K, N; };
    const Shape shapes[] = {{"wq_a 5120x1280", 5120, 1280}, {"wq_b 1280x32768", 1280, 32768}, {"wkv 5120x512", 5120, 512},
                            {"wo_a grp 4096x1024", 4096, 1024}, {"wo_b 8192x5120", 8192, 5120},
                            {"shexp gate 5120x2304", 5120, 2304}, {"shexp down 2304x5120", 2304, 5120},
                            {"head 5120x129280", 5120, 129280}};
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_F16};
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    for (const Shape& sh : shapes) {
        for (ggml_type t : types) {
            ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* W = ggml_new_tensor_2d(ctx, t, sh.K, sh.N);
            ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.K, ncols);
            ggml_tensor* Y = ggml_mul_mat(ctx, W, X);
            ggml_cgraph* g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            // weights: quantize random data on the host
            std::vector<float> wf((size_t) sh.K * sh.N);
            for (float& v : wf) v = ud(rng) * 0.05f;
            std::vector<uint8_t> wq(ggml_nbytes(W));
            ggml_quantize_chunk(t, wf.data(), wq.data(), 0, sh.N, sh.K, nullptr);
            ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
            std::vector<float> xf((size_t) sh.K * ncols);
            for (float& v : xf) v = ud(rng);
            ggml_backend_tensor_set(X, xf.data(), 0, xf.size() * 4);
            for (int w = 0; w < 10; ++w) ggml_backend_graph_compute(be, g);
            ggml_backend_synchronize(be);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
            ggml_backend_synchronize(be);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::printf("%-24s %-5s ncols %d: %8.1f us  %6.0f GB/s  (%.1f MB)\n", sh.name, ggml_type_name(t), ncols, us,
                        ggml_nbytes(W) / us / 1e3, ggml_nbytes(W) / 1e6);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
    return 0;
}
