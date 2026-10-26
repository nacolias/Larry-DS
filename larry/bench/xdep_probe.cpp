// xdep_probe.cpp - how long card 0 takes to start work that depends on card 1, three ways (larry, B70 x2):
//   event : card 1 copies 20 KB into card 0's memory; card 0's kernel depends_on that copy's event (the tier's sum)
//   host  : the host waits for the copy's event, then submits card 0's kernel
//   flag  : card 1's kernel sets a pinned host flag after the copy; card 0's kernel spins on it (bounded)
// Each stage ends with a one-lane kernel setting a pinned host flag; a monitor thread times the flags' arrival.
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 2) { std::printf("need 2 GPUs\n"); return 1; }
    sycl::context ctx(std::vector<sycl::device>{gpus[0], gpus[1]});
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order()), q1(ctx, gpus[1], sycl::property::queue::in_order());
    const size_t N = 5120;
    float* src = sycl::malloc_device<float>(N, q1);
    float* dst = sycl::malloc_device<float>(N, q0);
    int* dflag = sycl::malloc_host<int>(1, ctx);   // pinned host memory: card 1 writes it, card 0 polls it
    volatile int* hf = (volatile int*) sycl::malloc_host<int>(64, ctx);   // [0] card 1 done, [1] card 0 started
    q1.fill(src, 1.0f, N).wait();
    *dflag = 0;
    std::printf("p2p access 1->0: %d\n", (int) gpus[1].ext_oneapi_can_access_peer(gpus[0]));
    std::atomic<bool> stop{false};
    std::atomic<double> ta{0}, tb{0};
    std::thread mon([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            if (hf[0] && ta.load() == 0) ta = now_us();
            if (hf[1] && tb.load() == 0) tb = now_us();
        }
    });
    auto run = [&](const char* name, int mode) {
        double sum = 0, mx = 0; int n = 0;
        for (int it = 0; it < 300; ++it) {
            hf[0] = 0; hf[1] = 0; ta = 0; tb = 0;
            *(volatile int*) dflag = 0;
            // card 1: a ~20 us kernel, the copy into card 0, then the host flag
            q1.parallel_for(sycl::nd_range<1>(4096, 256), [=](sycl::nd_item<1> it) {
                float v = src[it.get_global_id(0) % N];
                for (int k = 0; k < 2000; ++k) v = v * 0.999f + 0.001f;
                if (v < -1e30f) src[0] = v;
            });
            sycl::event ec = q1.memcpy(dst, src, N * 4);
            if (mode == 2) q1.single_task([=] { sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                  sycl::access::address_space::global_space>(*dflag).store(1, sycl::memory_order::release); });
            sycl::event ef = q1.single_task([=] { hf[0] = 1; });
            if (mode == 0) {
                q0.submit([&](sycl::handler& h) { h.depends_on(ec); h.single_task([=] { hf[1] = 1; }); });
            } else if (mode == 1) {
                ec.wait();
                q0.single_task([=] { hf[1] = 1; });
            } else {
                q0.single_task([=] {
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space> f(*dflag);
                    for (int k = 0; k < (1 << 26) && f.load(sycl::memory_order::acquire) == 0; ++k) {}   // bounded: never a hung card
                    hf[1] = 1;
                });
            }
            q0.wait(); q1.wait();
            const double t0 = now_us();
            while ((ta.load() == 0 || tb.load() == 0) && now_us() - t0 < 1e5) {}
            if (it < 20) continue;
            const double d = tb.load() - ta.load();
            sum += d; mx = std::max(mx, d); ++n;
        }
        std::printf("%-6s card 1 done -> card 0 kernel ran: mean %.1f us, max %.1f us\n", name, sum / n, mx);
    };
    run("event", 0);
    run("host", 1);
    run("flag", 2);
    stop = true; mon.join();
    return 0;
}
