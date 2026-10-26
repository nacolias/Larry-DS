// p2p_kread.cpp - larry: the tier calls' X read (card 0 -> cards 1-3, 168 MB a card at 16K) by a KERNEL on each expert
// card (peer loads through the compute engine) instead of a queue memcpy (the card's one user copy engine, which then
// starves the prompt's staging DMAs: dma_load, pinned -> cards 1-3 28.4 -> 2.7 GB/s while the memcpys run).  Measures,
// with peer access enabled card k -> card 0: the kernel read alone, the pinned DMA alone, both at once; and checks the
// copied bytes.  Run with the service stopped: p2p_kread [seconds a test]
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const double T = argc > 1 ? std::atof(argv[1]) : 2.0;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 4) { std::fprintf(stderr, "need 4 cards\n"); return 1; }
    for (int k = 1; k < 4; ++k) {
        if (!devs[k].ext_oneapi_can_access_peer(devs[0])) { std::fprintf(stderr, "card %d cannot access card 0\n", k); return 1; }
        devs[k].ext_oneapi_enable_peer_access(devs[0]);
    }
    sycl::context ctx(devs);
    std::vector<sycl::queue> q, q2;
    for (auto& d : devs) { q.emplace_back(ctx, d, sycl::property::queue::in_order()); q2.emplace_back(ctx, d, sycl::property::queue::in_order()); }
    const size_t C = 1u << 20, PIN = 1536u << 20, DV = 512u << 20, X = 168u << 20;
    uint8_t* pin = sycl::malloc_host<uint8_t>(PIN, ctx);
    std::memset(pin, 7, PIN);
    std::vector<uint8_t*> dv(4), xr(4);
    for (int k = 1; k < 4; ++k) { dv[k] = sycl::malloc_device<uint8_t>(DV, q[k]); xr[k] = sycl::malloc_device<uint8_t>(X, q2[k]); }
    uint8_t* x0 = sycl::malloc_device<uint8_t>(X, q[0]);
    {
        std::vector<uint32_t> h(X / 4);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (uint32_t) (i * 2654435761u);
        q[0].memcpy(x0, h.data(), X).wait();
    }
    for (int k = 1; k < 4; ++k) q[k].memcpy(dv[k], pin, C).wait();
    // the kernel read: work-groups of 256, each work-item 16-byte loads in a grid-stride loop
    auto kread = [&](int k) {
        using v4 = sycl::vec<uint32_t, 4>;
        const v4* s = reinterpret_cast<const v4*>(x0);
        v4* d = reinterpret_cast<v4*>(xr[k]);
        const size_t n = X / 16, G = (size_t) 32 * 256 * 8;
        return q2[k].parallel_for(sycl::nd_range<1>(G, 256), [=](sycl::nd_item<1> it) {
            for (size_t i = it.get_global_id(0); i < n; i += G) d[i] = s[i];
        });
    };
    // correctness: one kernel read per card, compared on the host
    for (int k = 1; k < 4; ++k) kread(k).wait();
    {
        std::vector<uint32_t> a(X / 4), b(X / 4);
        q[0].memcpy(a.data(), x0, X).wait();
        for (int k = 1; k < 4; ++k) {
            q2[k].memcpy(b.data(), xr[k], X).wait();
            std::printf("card %d: kernel peer read %s\n", k, std::memcmp(a.data(), b.data(), X) ? "DIFFERS" : "matches card 0");
        }
    }
    auto dma = [&](double secs) {
        std::vector<std::vector<sycl::event>> ev(4, std::vector<sycl::event>(16));
        size_t n = 0, off = 0;
        const double t0 = now_s();
        while (now_s() - t0 < secs)
            for (int k = 1; k < 4; ++k) {
                sycl::event& e = ev[k][(n / 3) % 16];
                e.wait();
                e = q[k].memcpy(dv[k] + (n / 3 * C) % (DV - C), pin + off, C);
                off = (off + C) % (PIN - C);
                ++n;
            }
        for (int k = 1; k < 4; ++k) q[k].wait();
        return n * (double) C / (now_s() - t0) / 1e9;
    };
    for (int mode : {0, 1, 2}) {   // 0 kernel read alone, 1 DMA alone, 2 both
        std::atomic<bool> stop{false};
        std::atomic<int64_t> pb{0};
        std::thread th;
        double t0 = now_s();
        if (mode != 1)
            th = std::thread([&] {
                while (!stop.load()) {
                    for (int k = 1; k < 4; ++k) kread(k);
                    for (int k = 1; k < 4; ++k) q2[k].wait();
                    pb += 3 * (int64_t) X;
                }
            });
        double r = 0;
        if (mode != 0) r = dma(T);
        else std::this_thread::sleep_for(std::chrono::duration<double>(T));
        const double dt = now_s() - t0;
        stop = true;
        if (th.joinable()) th.join();
        std::printf("%-16s kernel peer reads %5.1f GB/s (card 0 -> 1-3), pinned -> cards 1-3 DMA %5.1f GB/s\n",
                    mode == 0 ? "kernel alone" : mode == 1 ? "DMA alone" : "both", pb.load() / dt / 1e9, r);
        std::fflush(stdout);
    }
    return 0;
}
