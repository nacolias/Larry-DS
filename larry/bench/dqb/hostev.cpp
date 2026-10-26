// hostev.cpp - larry: card 0's queue waits for a host-signaled Level Zero event (wrapped as a SYCL event and used in
// depends_on) that a CPU thread signals ~150 us after the submission - the decode / verify sum waiting for the
// CPU-pool experts without the host blocking.  Reports: how long the dependent submission blocks the host, and
// zeEventHostSignal -> the dependent kernel's stamp.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::context ctx(gpus);
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order());
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    std::vector<ze_device_handle_t> zd;
    for (auto& d : gpus) zd.push_back(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d));
    const int NE = 64;
    ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, (uint32_t) NE};
    ze_event_pool_handle_t pool = nullptr;
    if (zeEventPoolCreate(zc, &pd, (uint32_t) zd.size(), zd.data(), &pool) != ZE_RESULT_SUCCESS) { std::printf("pool failed\n"); return 1; }
    std::vector<ze_event_handle_t> ev(NE);
    for (int i = 0; i < NE; ++i) {
        ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, (uint32_t) i, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
        if (zeEventCreate(pool, &ed, &ev[i]) != ZE_RESULT_SUCCESS) { std::printf("event failed\n"); return 1; }
    }
    volatile int* stamp = (volatile int*) sycl::malloc_host<int>(16, ctx);
    float* buf = sycl::malloc_device<float>(1 << 20, q0);
    q0.fill(buf, 1.0f, 1 << 20).wait();
    std::atomic<double> t_gpu{0};
    std::atomic<bool> stop{false};
    std::thread mon([&] { while (!stop.load()) { if (stamp[0] && t_gpu.load() == 0) t_gpu = now_us(); } });
    std::vector<double> lat, sub;
    int bad = 0;
    for (int it = 0; it < 300; ++it) {
        ze_event_handle_t ze = ev[(size_t) (it % NE)];
        zeEventHostReset(ze);
        stamp[0] = 0; t_gpu = 0;
        sycl::event se = sycl::make_event<sycl::backend::ext_oneapi_level_zero>({ze, sycl::ext::oneapi::level_zero::ownership::keep}, ctx);
        q0.parallel_for(sycl::nd_range<1>(8192, 256), [=](sycl::nd_item<1> i) {
            float x = buf[i.get_global_id(0)];
            for (int r = 0; r < 1800; ++r) x = x * 0.999f + 0.001f;
            if (x < -1e30f) buf[0] = x;
        });
        double t_sig = 0;
        std::thread cpu([&] {   // started first: a submit that blocks on the event still finishes
            const double t0 = now_us();
            while (now_us() - t0 < 150) {}
            t_sig = now_us();
            zeEventHostSignal(ze);
        });
        const double ts0 = now_us();
        q0.submit([&](sycl::handler& h) { h.depends_on(se); h.single_task([=] { stamp[0] = 1; }); });
        const double ts1 = now_us();
        cpu.join();
        q0.wait();
        const double t0 = now_us();
        while (t_gpu.load() == 0 && now_us() - t0 < 1e5) {}
        if (t_gpu.load() == 0) { ++bad; continue; }
        if (it >= 20) { lat.push_back(t_gpu.load() - t_sig); sub.push_back(ts1 - ts0); }
    }
    stop = true;
    mon.join();
    std::printf("bad %d, samples %zu\n", bad, lat.size());
    if (lat.empty()) return 2;
    std::sort(lat.begin(), lat.end());
    std::sort(sub.begin(), sub.end());
    std::printf("dependent submit blocks the host: median %.1f us, max %.1f | signal -> stamp: median %.1f us p10 %.1f p90 %.1f max %.1f\n",
                sub[sub.size() / 2], sub.back(), lat[lat.size() / 2], lat[lat.size() / 10], lat[lat.size() * 9 / 10], lat.back());
    return 0;
}
