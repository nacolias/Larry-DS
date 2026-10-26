// sum_lat.cpp - larry: the decode sum's dependency wait, engine-shaped.  Card 0 runs a ~80 us kernel (the shared
// expert), then stamp P, then the sum kernel that depends_on the three expert cards' copies, then stamp B.  Cards 1-3
// each wait on a spin flag (armed: released by the host), run a ~50 us kernel, copy 20 KB into card 0 (that copy's
// event is the sum's dependency), then stamp C.  The sum and its dependencies are queued BEFORE the work runs, like
// the engine (the host is ~50 us ahead).  Reported: B - max(P, C1..C3).
//   mode 0: depends_on the copies' events (the engine)
//   mode 1: the copies end with a kernel setting a pinned host flag; card 0's sum spins on the three flags
//   mode 2: like 0, but the sum is submitted only after every copy is done (host waits: the probe's earlier shape)
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <cstdlib>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int NK = argc > 1 ? std::atoi(argv[1]) : 3;
    const int MODE = argc > 2 ? std::atoi(argv[2]) : -1;
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 4) { std::printf("need 4 GPUs\n"); return 1; }
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto& d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t N = 5120;
    std::vector<float*> buf;
    for (auto& qq : q) buf.push_back(sycl::malloc_device<float>(N * 4, qq));
    volatile int* hf = (volatile int*) sycl::malloc_host<int>(64, ctx);   // [0] P, [1] B, [2..4] C1..C3
    int* rel = sycl::malloc_host<int>(64, ctx);                           // [k] release flags, [16 + k] copy-done flags
    for (size_t k = 0; k < q.size(); ++k) q[k].fill(buf[k], 1.0f, N * 4).wait();
    std::atomic<bool> stop{false};
    std::atomic<double> tt[8];
    std::thread mon([&] {
        while (!stop.load(std::memory_order_relaxed))
            for (int i = 0; i < 5; ++i)
                if (hf[i] && tt[i].load() == 0) tt[i] = now_us();
    });
    auto busy = [](sycl::queue& qq, float* p, int iters) {
        qq.parallel_for(sycl::nd_range<1>(8192, 256), [=](sycl::nd_item<1> i) {
            float x = p[i.get_global_id(0) % 5120];
            for (int r = 0; r < iters; ++r) x = x * 0.999f + 0.001f;
            if (x < -1e30f) p[0] = x;
        });
    };
    const char* names[] = { "depends_on copy events (engine)", "sum spins on host flags", "host waits, then submits sum", "depends_on, host idle 2 ms" };
    for (int mode = 0; mode < 4; ++mode) {
        if (MODE >= 0 && mode != MODE) continue;
        std::vector<double> v, vp, vc;
        for (int it = 0; it < 300; ++it) {
            for (int i = 0; i < 8; ++i) { hf[i] = 0; tt[i] = 0; }
            for (int i = 0; i < 64; ++i) ((volatile int*) rel)[i] = 0;
            std::vector<sycl::event> cp;
            for (int k = 1; k <= NK; ++k) {   // armed cards: spin on the release, then the work, the copy, stamp C
                busy(q[k], buf[k], 1500);
                cp.push_back(q[k].memcpy(buf[0] + (size_t) k * N, buf[k], N * 4));
                if (mode == 1) {
                    int* df = rel + 16 + k;
                    q[k].single_task([=] { sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                            sycl::access::address_space::global_space>(*df).store(1, sycl::memory_order::release); });
                }
                q[k].single_task([=] { hf[1 + k] = 1; });
            }
            busy(q[0], buf[0], 2400);   // the shared expert (~80 us)
            q[0].single_task([=] { hf[0] = 1; });
            float* o = buf[0];
            if (mode == 2) for (auto& e : cp) e.wait();
            if (mode == 1) {
                int* df = rel + 16;
                q[0].parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    for (int k = 1; k <= NK; ++k) {
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                         sycl::access::address_space::global_space> f(df[k]);
                        for (int r = 0; r < (1 << 18) && f.load(sycl::memory_order::acquire) == 0; ++r) {}
                    }
                    o[i] = o[N + i] + (NK >= 2 ? o[2 * N + i] : 0.0f) + (NK >= 3 ? o[3 * N + i] : 0.0f);
                });
            } else {
                q[0].submit([&](sycl::handler& h) {
                    h.depends_on(cp);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { o[i] = o[N + i] + (NK >= 2 ? o[2 * N + i] : 0.0f) + (NK >= 3 ? o[3 * N + i] : 0.0f); });
                });
            }
            q[0].single_task([=] { hf[1] = 1; });
            if (mode == 3) { const double ti = now_us(); while (now_us() - ti < 2000) {} }
            {
                const double tw = now_us();
                bool ok = false;
                while (now_us() - tw < 3e6) {
                    bool all = true;
                    for (auto& e : cp) if (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) all = false;
                    if (all) { ok = true; break; }
                }
                if (!ok) {
                    std::printf("mode %d it %d: copies not done after 3 s:", mode, it);
                    for (auto& e : cp) std::printf(" %d", (int) (e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete));
                    std::printf(" | stamps P %d B %d C %d %d %d\n", hf[0], hf[1], hf[2], hf[3], hf[4]);
                    std::fflush(stdout);
                    std::_Exit(3);
                }
            }
            for (auto& qq : q) qq.wait();
            const double t0 = now_us();
            while ((tt[0].load() == 0 || tt[1].load() == 0 || (NK >= 1 && tt[2].load() == 0) || (NK >= 2 && tt[3].load() == 0) || (NK >= 3 && tt[4].load() == 0)) && now_us() - t0 < 1e5) {}
            if (it < 30) continue;
            double c = 0; for (int k = 1; k <= NK; ++k) c = std::max(c, tt[1 + k].load());
            v.push_back(tt[1].load() - std::max(c, tt[0].load()));
            vp.push_back(tt[1].load() - tt[0].load());
            vc.push_back(tt[1].load() - c);
        }
        std::sort(v.begin(), v.end());
        std::sort(vp.begin(), vp.end());
        std::sort(vc.begin(), vc.end());
        std::printf("%-34s B - max(P, C): median %6.1f us p90 %6.1f | B - P median %6.1f | B - max C median %6.1f\n", names[mode],
                    v[v.size() / 2], v[v.size() * 9 / 10], vp[vp.size() / 2], vc[vc.size() / 2]);
    }
    stop = true;
    mon.join();
    return 0;
}
