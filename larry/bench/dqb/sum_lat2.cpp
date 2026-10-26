// sum_lat2.cpp - larry: the decode sum's wait for the expert cards when they finish just AFTER card 0 reaches it
// (the engine's race).  Card 0: a busy kernel (k0 iterations, the shared expert), stamp P, the sum, stamp B.  Cards
// 1-3: a busy kernel (kk iterations), a 20 KB copy into card 0, then stamp C.
//   mode 0: the sum depends_on the copies' events (the engine today)
//   mode 1: each card also copies a 4-byte sequence number into a flag word in card 0's memory after its data; the
//           sum kernel spins (bounded) on those local flags - no cross-device event
// usage: sum_lat2 k0 kk mode [extra_queues_on_card0]
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int k0 = argc > 1 ? std::atoi(argv[1]) : 2400, kk = argc > 2 ? std::atoi(argv[2]) : 2600;
    const int mode = argc > 3 ? std::atoi(argv[3]) : 0, extra = argc > 4 ? std::atoi(argv[4]) : 0;
    const int behind = argc > 5 ? std::atoi(argv[5]) : 0;   // small kernels queued behind the sum (the engine's next graphs)
    const int dirty_mb = argc > 6 ? std::atoi(argv[6]) : 0;   // MiB card 0 writes right before stamp P (dirty L3)
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 4) { std::printf("need 4 GPUs\n"); return 1; }
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto& d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    std::vector<sycl::queue> xq;   // extra queues on card 0 (each runs one tiny kernel first: live contexts)
    for (int i = 0; i < extra; ++i) { xq.emplace_back(ctx, gpus[0], sycl::property::queue::in_order()); xq.back().single_task([] {}).wait(); }
    const size_t N = 5120;
    std::vector<float*> buf;
    for (auto& qq : q) buf.push_back(sycl::malloc_device<float>(N * 4, qq));
    int* flags = sycl::malloc_device<int>(16, q[0]);   // card 0: [k] = the last sequence number card k delivered
    int* seqsrc = sycl::malloc_host<int>(64, ctx);     // pinned sources of the flag copies
    q[0].memset(flags, 0, 16 * sizeof(int)).wait();
    volatile int* hf = (volatile int*) sycl::malloc_host<int>(64, ctx);
    for (size_t k = 0; k < q.size(); ++k) q[k].fill(buf[k], 1.0f, N * 4).wait();
    std::atomic<bool> stop{false};
    std::atomic<double> tt[8];
    for (auto& t : tt) t = 0;
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
    int* errp = sycl::malloc_host<int>(1, ctx);
    float* dbuf = sycl::malloc_device<float>((size_t) 64 << 18, q[0]);   // 64 MiB
    *errp = 0;
    std::vector<double> v, vc, vp;
    for (int it = 0; it < 330; ++it) {
        for (int i = 0; i < 8; ++i) { hf[i] = 0; tt[i] = 0; }
        const int seq = it + 1;
        std::vector<sycl::event> cp;
        for (int k = 1; k <= 3; ++k) {
            busy(q[k], buf[k], kk);
            cp.push_back(q[k].memcpy(buf[0] + (size_t) k * N, buf[k], N * 4));
            if (mode >= 1) {
                seqsrc[16 * (k - 1) + (it & 1)] = seq;   // a slot not in flight (two per card, alternating)
                q[k].memcpy(flags + k, seqsrc + 16 * (k - 1) + (it & 1), sizeof(int));
            }
            q[k].single_task([=] { hf[1 + k] = 1; });
        }
        busy(q[0], buf[0], k0);
        if (dirty_mb > 0) {   // write dirty_mb MiB (the shared expert's outputs, attention scratch...)
            const size_t nf = (size_t) dirty_mb << 18;
            float* db = dbuf;
            q[0].parallel_for(sycl::range<1>(nf), [=](sycl::id<1> i) { db[i] = db[i] * 0.5f + 1.0f; });
        }
        q[0].single_task([=] { hf[0] = 1; });
        float* o = buf[0];
        if (mode == 2) {   // one lane waits for the three flags, then the plain sum
            int* fl = flags;
            q[0].single_task([=] {
                for (int k = 1; k <= 3; ++k) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space> f(fl[k]);
                    int r = 0;
                    for (; r < (1 << 24) && f.load(sycl::memory_order::acquire) < seq; ++r) {}
                    if (r == (1 << 24)) *errp = 1;
                }
            });
            q[0].parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { o[i] = o[N + i] + o[2 * N + i] + o[3 * N + i]; });
        } else if (mode == 1) {
            int* fl = flags;
            q[0].parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                for (int k = 1; k <= 3; ++k) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                     sycl::access::address_space::global_space> f(fl[k]);
                    int r = 0;
                    for (; r < (1 << 22) && f.load(sycl::memory_order::acquire) < seq; ++r) {}
                    if (r == (1 << 22)) *errp = 1;
                }
                o[i] = o[N + i] + o[2 * N + i] + o[3 * N + i];
            });
        } else {
            q[0].submit([&](sycl::handler& h) {
                h.depends_on(cp);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { o[i] = o[N + i] + o[2 * N + i] + o[3 * N + i]; });
            });
        }
        q[0].single_task([=] { hf[1] = 1; });
        for (int b = 0; b < behind; ++b) q[0].parallel_for(sycl::range<1>(5120), [=](sycl::id<1> i) { o[i] = o[i] * 0.5f + 1.0f; });
        for (int k = 3; k >= 0; --k) q[k].wait();
        const double t0 = now_us();
        while ((tt[0].load() == 0 || tt[1].load() == 0 || tt[2].load() == 0 || tt[3].load() == 0 || tt[4].load() == 0) && now_us() - t0 < 1e5) {}
        if (it < 30) continue;
        const double c = std::max({ tt[2].load(), tt[3].load(), tt[4].load() });
        v.push_back(tt[1].load() - std::max(c, tt[0].load()));
        vc.push_back(c - tt[0].load());
        vp.push_back(tt[1].load() - tt[0].load());
    }
    stop = true;
    mon.join();
    std::sort(v.begin(), v.end());
    std::sort(vc.begin(), vc.end());
    std::sort(vp.begin(), vp.end());
    std::printf("k0 %d kk %d mode %d extra %d: B - max(P,C) median %6.1f us p10 %6.1f p90 %6.1f | C - P median %6.1f | B - P median %6.1f%s\n",
                k0, kk, mode, extra, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], vc[vc.size() / 2], vp[vp.size() / 2],
                *errp ? "  SPIN TIMEOUT" : "");
    return 0;
}
