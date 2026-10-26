// launch_probe.cpp - what a small kernel costs on the B70 when kernels run back to back: an in-order queue (each kernel
// waits for the previous), the same sequence recorded once into a SYCL graph and replayed, and an in-order queue with
// the kernels' GPU timestamps (the gap between one kernel's end and the next one's start).
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace sycl_ex = sycl::ext::oneapi::experimental;

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2000;
    sycl::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t n = 5120;
    float* a = sycl::malloc_device<float>(n, q);
    q.memset(a, 0, n * 4).wait();
    auto k = [&](sycl::queue& qq) {
        return qq.parallel_for(sycl::nd_range<1>(n, 256), [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1.0f; });
    };
    for (int i = 0; i < 50; ++i) k(q);
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) k(q);
    q.wait();
    const double us_inorder = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
    // GPU timestamps: gaps between consecutive kernels
    std::vector<sycl::event> ev(200);
    for (auto& e : ev) e = k(q);
    q.wait();
    double busy = 0, span = 0;
    for (size_t i = 0; i < ev.size(); ++i) {
        const auto s = ev[i].get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto e = ev[i].get_profiling_info<sycl::info::event_profiling::command_end>();
        busy += (e - s) / 1000.0;
        if (i + 1 == ev.size()) span = (e - ev[0].get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0;
    }
    std::printf("in-order queue: %.2f us per kernel (host-timed); GPU: kernel %.2f us, start-to-start %.2f us\n", us_inorder,
                busy / ev.size(), span / ev.size());
    // the same kernels recorded once into a graph, replayed
    {
        sycl::queue qg(q.get_context(), q.get_device(), sycl::property::queue::in_order());
        sycl_ex::command_graph g(qg);
        const int G = 100;
        g.begin_recording(qg);
        for (int i = 0; i < G; ++i) k(qg);
        g.end_recording();
        auto ex = g.finalize();
        for (int i = 0; i < 3; ++i) qg.ext_oneapi_graph(ex);
        qg.wait();
        const int R = N / G;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) qg.ext_oneapi_graph(ex);
        qg.wait();
        const double us_g = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (R * G);
        std::printf("graph replay (%d kernels per graph): %.2f us per kernel\n", G, us_g);
    }
    // one queue per kernel chain is not the model; an out-of-order queue with independent kernels: pure launch cost
    {
        sycl::queue qo(q.get_context(), q.get_device());
        for (int i = 0; i < 50; ++i) k(qo);
        qo.wait();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) k(qo);
        qo.wait();
        std::printf("out-of-order, independent: %.2f us per kernel\n",
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N);
    }
    return 0;
}
