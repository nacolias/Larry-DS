// clk_probe.cpp - engine1: is sycl_ext_oneapi_clock's device clock usable for in-graph timestamps on the B70?
// stamp kernel | busy kernel (~T us, host-timed alone first) | stamp kernel, eager and in a replayed graph: the
// ticks between the stamps vs the busy kernel's host time give the clock's rate and whether two single-task kernels
// (any EU) read one consistent clock.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const auto dev = q.get_device();
    std::printf("%s: clock device %d, work_group %d, sub_group %d\n", dev.get_info<sycl::info::device::name>().c_str(),
                (int) dev.has(sycl::aspect::ext_oneapi_clock_device), (int) dev.has(sycl::aspect::ext_oneapi_clock_work_group),
                (int) dev.has(sycl::aspect::ext_oneapi_clock_sub_group));
    if (!dev.has(sycl::aspect::ext_oneapi_clock_sub_group)) return 1;
    uint64_t * st = sycl::malloc_device<uint64_t>(64, q);
    float * buf = sycl::malloc_device<float>(1 << 20, q);
    q.memset(buf, 0, (1 << 20) * 4).wait();
    auto stamp = [&](int i) { q.single_task([=] { st[i] = sx::clock<sx::clock_scope::sub_group>(); }); };
    auto busy = [&](int iters) {   // 256 work-groups of dependent FMAs: a fixed GPU time
        q.parallel_for(sycl::nd_range<1>(256 * 256, 256), [=](sycl::nd_item<1> it) {
            float v = buf[it.get_global_id(0)];
            for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1e-7f;
            buf[it.get_global_id(0)] = v;
        });
    };
    for (int iters : {20000, 100000, 400000}) {
        busy(iters); q.wait();
        const double t0 = now_us();
        for (int r = 0; r < 10; ++r) busy(iters);
        q.wait();
        const double tb = (now_us() - t0) / 10;
        // eager: stamp, busy, stamp (x8)
        for (int r = 0; r < 8; ++r) { stamp(2 * r); busy(iters); stamp(2 * r + 1); }
        q.wait();
        std::vector<uint64_t> h(64);
        q.memcpy(h.data(), st, 64 * 8).wait();
        double mn = 1e30, mx = 0;
        for (int r = 0; r < 8; ++r) { const double d = (double) (h[2 * r + 1] - h[2 * r]); mn = std::min(mn, d); mx = std::max(mx, d); }
        // graph: the same 8 triples recorded once, replayed
        sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), dev);
        g.begin_recording(q);
        for (int r = 0; r < 8; ++r) { stamp(16 + 2 * r); busy(iters); stamp(16 + 2 * r + 1); }
        g.end_recording(q);
        auto ge = g.finalize();
        q.ext_oneapi_graph(ge).wait();
        q.memcpy(h.data(), st, 64 * 8).wait();
        double gmn = 1e30, gmx = 0;
        for (int r = 0; r < 8; ++r) { const double d = (double) (h[16 + 2 * r + 1] - h[16 + 2 * r]); gmn = std::min(gmn, d); gmx = std::max(gmx, d); }
        // back-to-back stamps: the cost of a stamp kernel in a graph
        sx::command_graph<sx::graph_state::modifiable> g2(q.get_context(), dev);
        g2.begin_recording(q);
        for (int i = 0; i < 32; ++i) stamp(32 + i);
        g2.end_recording(q);
        auto ge2 = g2.finalize();
        q.ext_oneapi_graph(ge2).wait();
        q.memcpy(h.data(), st, 64 * 8).wait();
        const double step = (double) (h[63] - h[32]) / 31;
        std::printf("busy %6d iters: host %8.1f us | eager ticks %.0f..%.0f (%.4f..%.4f ticks/us) | graph ticks %.0f..%.0f (%.4f ticks/us) | stamp-to-stamp %.1f ticks\n",
                    iters, tb, mn, mx, mn / tb, mx / tb, gmn, gmx, gmn / tb, step);
    }
    sycl::free(st, q); sycl::free(buf, q);
    return 0;
}
