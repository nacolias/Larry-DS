// submit_bench.cpp - what one small kernel costs to SUBMIT on larry (host side), three ways:
//   direct    : q.parallel_for (an event per kernel, what ggml-sycl does)
//   enqueue   : sycl ext enqueue functions (no event)
//   graph     : the same N kernels recorded once into a SYCL graph, replayed
//   record    : record + update + replay every time (what ggml-sycl's graph mode does)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

namespace syclex = sycl::ext::oneapi::experimental;

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2000;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    float* a = sycl::malloc_device<float>(1 << 16, q);
    q.memset(a, 0, 4 << 16).wait();
    auto kern = [=](sycl::nd_item<1> it) { const size_t i = it.get_global_id(0); a[i] = a[i] * 0.5f + 1.0f; };
    const sycl::nd_range<1> nr(sycl::range<1>(4096), sycl::range<1>(256));
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto us = [](auto t0, auto t1) { return std::chrono::duration<double, std::micro>(t1 - t0).count(); };
    for (int w = 0; w < 3; ++w) { for (int i = 0; i < N; ++i) q.parallel_for(nr, kern); q.wait(); }
    auto t0 = now();
    for (int i = 0; i < N; ++i) q.parallel_for(nr, kern);
    auto t1 = now();
    q.wait();
    auto t2 = now();
    std::printf("direct : submit %.2f us/kernel, total %.2f us/kernel\n", us(t0, t1) / N, us(t0, t2) / N);
    for (int w = 0; w < 3; ++w) { for (int i = 0; i < N; ++i) syclex::nd_launch(q, nr, kern); q.wait(); }
    t0 = now();
    for (int i = 0; i < N; ++i) syclex::nd_launch(q, nr, kern);
    t1 = now();
    q.wait();
    t2 = now();
    std::printf("enqueue: submit %.2f us/kernel, total %.2f us/kernel\n", us(t0, t1) / N, us(t0, t2) / N);
    // graph: record once
    syclex::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int i = 0; i < N; ++i) q.parallel_for(nr, kern);
    g.end_recording();
    auto ge = g.finalize(syclex::property::graph::updatable{});
    for (int w = 0; w < 3; ++w) { q.ext_oneapi_graph(ge); q.wait(); }
    t0 = now();
    for (int r = 0; r < 10; ++r) q.ext_oneapi_graph(ge);
    t1 = now();
    q.wait();
    t2 = now();
    std::printf("graph  : submit %.3f us/kernel, total %.2f us/kernel (replay of %d)\n", us(t0, t1) / (10.0 * N), us(t0, t2) / (10.0 * N), N);
    // record + update + replay each time
    t0 = now();
    for (int r = 0; r < 5; ++r) {
        syclex::command_graph g2(q.get_context(), q.get_device());
        g2.begin_recording(q);
        for (int i = 0; i < N; ++i) q.parallel_for(nr, kern);
        g2.end_recording();
        ge.update(g2);
        q.ext_oneapi_graph(ge);
    }
    q.wait();
    t1 = now();
    std::printf("record : %.2f us/kernel (record + update + replay)\n", us(t0, t1) / (5.0 * N));
    // GPU-only time of one kernel (replay of the graph measures it best): total above
    return 0;
}
