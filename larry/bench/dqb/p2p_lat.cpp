// p2p_lat.cpp - larry: what the decode sum waits for after an expert card's kernels end.  Card k runs a ~20 us
// kernel, then stamp A (a one-lane kernel setting a pinned host flag); then one of:
//   0 copy-k  : card k's queue copies 20 KB into card 0's memory; card 0's stamp B kernel depends_on that copy
//   1 dep     : no copy; card 0's stamp B depends_on card k's stamp A kernel (the bare cross-card dependency)
//   2 d2h     : card k's queue copies 20 KB into pinned host memory; card 0's stamp B depends_on it
//   3 pull-0  : card 0's queue copies the 20 KB from card k's memory (depends_on stamp A), then stamp B
// A monitor thread times the flags' arrival (host clock).  All four cards in one context, like the tier.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 2) { std::printf("need 2 GPUs\n"); return 1; }
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto& d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t N = 5120;
    std::vector<float*> buf;
    for (auto& qq : q) buf.push_back(sycl::malloc_device<float>(N, qq));
    float* hbuf = sycl::malloc_host<float>(N, ctx);
    volatile int* hf = (volatile int*) sycl::malloc_host<int>(64, ctx);
    for (size_t k = 0; k < q.size(); ++k) q[k].fill(buf[k], 1.0f, N).wait();
    std::atomic<bool> stop{false};
    std::atomic<double> ta{0}, tb{0};
    std::thread mon([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            if (hf[0] && ta.load() == 0) ta = now_us();
            if (hf[1] && tb.load() == 0) tb = now_us();
        }
    });
    const char* names[] = { "copy-k (card k -> card 0)", "dep only (no copy)", "d2h (card k -> pinned)", "pull-0 (card 0 copies)" };
    for (size_t k = 1; k < q.size(); ++k) {
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<double> v;
            for (int it = 0; it < 400; ++it) {
                hf[0] = 0; hf[1] = 0; ta = 0; tb = 0;
                float* src = buf[k];
                q[k].parallel_for(sycl::nd_range<1>(4096, 256), [=](sycl::nd_item<1> i) {
                    float x = src[i.get_global_id(0) % N];
                    for (int r = 0; r < 2000; ++r) x = x * 0.999f + 0.001f;
                    if (x < -1e30f) src[0] = x;
                });
                sycl::event ea = q[k].single_task([=] { hf[0] = 1; });
                if (mode == 0) {
                    sycl::event ec = q[k].memcpy(buf[0], src, N * 4);
                    q[0].submit([&](sycl::handler& h) { h.depends_on(ec); h.single_task([=] { hf[1] = 1; }); });
                } else if (mode == 1) {
                    q[0].submit([&](sycl::handler& h) { h.depends_on(ea); h.single_task([=] { hf[1] = 1; }); });
                } else if (mode == 2) {
                    sycl::event ec = q[k].memcpy(hbuf, src, N * 4);
                    q[0].submit([&](sycl::handler& h) { h.depends_on(ec); h.single_task([=] { hf[1] = 1; }); });
                } else {
                    q[0].submit([&](sycl::handler& h) { h.depends_on(ea); h.memcpy(buf[0], src, N * 4); });
                    q[0].single_task([=] { hf[1] = 1; });
                }
                q[0].wait();
                q[k].wait();
                const double t0 = now_us();
                while ((ta.load() == 0 || tb.load() == 0) && now_us() - t0 < 1e5) {}
                if (it >= 50 && ta.load() > 0 && tb.load() > 0) v.push_back(tb.load() - ta.load());
            }
            std::sort(v.begin(), v.end());
            double s = 0;
            for (double x : v) s += x;
            std::printf("card %zu -> 0  %-28s  A->B median %6.1f us  mean %6.1f  p10 %6.1f  p90 %6.1f\n", k, names[mode],
                        v[v.size() / 2], s / v.size(), v[v.size() / 10], v[v.size() * 9 / 10]);
        }
    }
    stop = true;
    mon.join();
    return 0;
}
