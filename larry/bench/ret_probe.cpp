// ret_probe.cpp - the decode tier's return path in the engine's context (the platform default context, 4 GPUs):
// card k: work kernel, stamp A, 20 KB copy into card 0's slot; card 0: a kernel depending on that copy, stamp B.
// modes: 0 card 0 kernel depends_on the copy event (the tier today); 1 card 0's queue pulls the copy itself after
// depends_on card k's work event; 2 the host waits for the copy event, then submits card 0's kernel.
// Stamps: one-lane kernels setting pinned host words that a monitor thread timestamps (the STRATA_GPU_TL scheme).
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> devs;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    sycl::context ctx = devs[0].get_platform().khr_get_default_context();
    std::vector<sycl::queue> q;
    for (auto& d : devs) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t N = 5120;
    float* slot = sycl::malloc_device<float>(4 * N, q[0]);
    std::vector<float*> part(devs.size());
    for (size_t k = 1; k < devs.size(); ++k) { part[k] = sycl::malloc_device<float>(N, q[k]); q[k].fill(part[k], 1.0f, N).wait(); }
    volatile int* hf = (volatile int*) sycl::malloc_host<int>(1024, ctx);   // [0] A (card 1 done), [64] B (card 0 ran)
    std::atomic<bool> stop{false};
    std::atomic<double> ta{0}, tb{0};
    std::thread mon([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            if (hf[0] && ta.load() == 0) ta = now_us();
            if (hf[64] && tb.load() == 0) tb = now_us();
        }
    });
    const int k = 2;   // an expert card behind the switch
    auto run = [&](const char* name, int mode) {
        std::vector<double> lat;
        for (int it = 0; it < 400; ++it) {
            hf[0] = 0; hf[64] = 0; ta = 0; tb = 0;
            float* src = part[k];
            float* dst = slot + (size_t) k * N;
            sycl::event ew = q[k].parallel_for(sycl::nd_range<1>(4096, 256), [=](sycl::nd_item<1> i) {
                float v = src[i.get_global_id(0) % N];
                for (int r = 0; r < 2000; ++r) v = v * 0.999f + 0.001f;
                if (v < -1e30f) src[0] = v;
            });
            sycl::event ea = q[k].single_task([=] { hf[0] = 1; });
            sycl::event ec;
            if (mode != 1) ec = q[k].memcpy(dst, src, N * 4);
            if (mode == 0) {
                q[0].submit([&](sycl::handler& h) { h.depends_on(ec); h.single_task([=] { hf[64] = 1; }); });
            } else if (mode == 1) {
                q[0].submit([&](sycl::handler& h) { h.depends_on(ea); h.memcpy(dst, src, N * 4); });
                q[0].single_task([=] { hf[64] = 1; });
            } else {
                ec.wait();
                q[0].single_task([=] { hf[64] = 1; });
            }
            q[0].wait(); q[k].wait();
            const double t0 = now_us();
            while ((ta.load() == 0 || tb.load() == 0) && now_us() - t0 < 1e5) {}
            if (it >= 30) lat.push_back(tb.load() - ta.load());
        }
        std::sort(lat.begin(), lat.end());
        std::printf("%-26s card %d done -> card 0 ran: p10 %.1f  p50 %.1f  p90 %.1f us\n", name, k, lat[lat.size() / 10],
                    lat[lat.size() / 2], lat[lat.size() * 9 / 10]);
    };
    run("depends_on the copy event", 0);
    run("card 0 pulls the part", 1);
    run("host waits, then submits", 2);
    stop = true;
    mon.join();
    return 0;
}
