// cpuflag.cpp - larry: can a one-lane kernel on card 0 wait for a flag a CPU thread writes into pinned host memory
// (the decode sum waiting for the CPU-pool experts without the host blocking)?  Per iteration: card 0 runs a ~60 us
// busy kernel, then the wait kernel (bounded spin on the host flag), then a stamp kernel; a CPU thread writes the flag
// ~150 us after the submission.  Reported: GPU stamp time - CPU write time.  Bounded everywhere (no hung cards).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdlib>
#include <immintrin.h>
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
    sycl::queue q0b(ctx, gpus[0], sycl::property::queue::in_order());   // a second queue on card 0 (like relq)
    int* flag = nullptr;
    if (std::getenv("UNCACHED")) {   // Level Zero host allocation the GPU reads uncached
        ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
        ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(zc, &hd, 4096, 64, &p) != ZE_RESULT_SUCCESS) { std::printf("zeMemAllocHost failed\n"); return 1; }
        flag = (int*) p;
        std::printf("flag: zeMemAllocHost BIAS_UNCACHED\n");
    } else {
        flag = sycl::malloc_host<int>(16, ctx);
    }
    int* err = sycl::malloc_host<int>(1, ctx);
    volatile int* stamp = (volatile int*) sycl::malloc_host<int>(16, ctx);
    float* buf = sycl::malloc_device<float>(1 << 20, q0);
    q0.fill(buf, 1.0f, 1 << 20).wait();
    *err = 0;
    for (int i = 0; i < 16; ++i) { flag[i] = 0; stamp[i] = 0; }
    std::atomic<double> t_gpu{0};
    std::atomic<bool> stop{false};
    std::thread mon([&] { while (!stop.load()) { if (stamp[0] && t_gpu.load() == 0) t_gpu = now_us(); } });
    std::vector<double> lat;
    int timeouts = 0;
    for (int it = 1; it <= 300; ++it) {
        stamp[0] = 0; t_gpu = 0;
        q0.parallel_for(sycl::nd_range<1>(8192, 256), [=](sycl::nd_item<1> i) {
            float x = buf[i.get_global_id(0)];
            for (int r = 0; r < 1800; ++r) x = x * 0.999f + 0.001f;
            if (x < -1e30f) buf[0] = x;
        });
        int* f = flag;
        int* ep = err;
        const int want = it;
        q0.single_task([=] {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system,
                             sycl::access::address_space::global_space> a(*f);
            int r = 0;
            for (; r < (1 << 20) && a.load(sycl::memory_order::acquire) < want; ++r) {}
            if (r == (1 << 20)) *ep = 1;
        });
        q0.single_task([=] { stamp[0] = 1; });
        if (it % 3 == 0) q0b.single_task([] {});   // some other card-0 queue traffic meanwhile
        // the CPU thread: ~150 us of "expert work", then the flag
        double t_cpu = 0;
        std::thread cpu([&] {
            const double t0 = now_us();
            while (now_us() - t0 < 150) {}
            std::atomic_thread_fence(std::memory_order_release);
            t_cpu = now_us();
            __atomic_store_n(f, want, __ATOMIC_RELEASE);
            if (std::getenv("FLUSH")) { _mm_clflush((const void*) f); _mm_mfence(); }   // the GPU may read DRAM without snooping
        });
        cpu.join();
        q0.wait();
        q0b.wait();
        const double t0 = now_us();
        while (t_gpu.load() == 0 && now_us() - t0 < 1e5) {}
        if (*err) { ++timeouts; *err = 0; if (timeouts >= 3) break; continue; }
        if (it > 20) lat.push_back(t_gpu.load() - t_cpu);
    }
    stop = true;
    mon.join();
    std::printf("timeouts %d, samples %zu\n", timeouts, lat.size());
    if (lat.empty()) return 2;
    std::sort(lat.begin(), lat.end());
    std::printf("CPU flag write -> GPU stamp: median %.1f us, p10 %.1f, p90 %.1f, max %.1f; spin timeouts %d of 300\n",
                lat[lat.size() / 2], lat[lat.size() / 10], lat[lat.size() * 9 / 10], lat.back(), timeouts);
    return 0;
}
