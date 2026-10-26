// dma_load.cpp - larry: is the prompt's staging held by DRAM or by the cards' copy engines?  Pinned host memory DMA'd
// onto cards 1-3 (1 MiB chunks round-robin, a queue a card, 16 in flight a card) for 2 s, against background load:
//   cpu N   N threads memcpy'ing pageable buffers (256 MiB each, AVX2 plain stores - the stagers' copy) = DRAM traffic
//   p2p     cards 1-3 each reading 168 MB from card 0 in a loop on a second queue (the tier calls' X reads)
// prints the DMA GB/s and the load's own rate.  Run with the service stopped: dma_load [seconds a test]
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__attribute__((target("avx2"))) static void copy_avx(uint8_t* d, const uint8_t* s, size_t n) {
    for (size_t i = 0; i + 128 <= n; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i*) (s + i)), b = _mm256_loadu_si256((const __m256i*) (s + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i*) (s + i + 64)), e = _mm256_loadu_si256((const __m256i*) (s + i + 96));
        _mm256_storeu_si256((__m256i*) (d + i), a); _mm256_storeu_si256((__m256i*) (d + i + 32), b);
        _mm256_storeu_si256((__m256i*) (d + i + 64), c); _mm256_storeu_si256((__m256i*) (d + i + 96), e);
    }
}

int main(int argc, char** argv) {
    const double T = argc > 1 ? std::atof(argv[1]) : 2.0;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 4) { std::fprintf(stderr, "need 4 cards\n"); return 1; }
    sycl::context ctx(devs);
    std::vector<sycl::queue> q, q2;
    for (auto& d : devs) { q.emplace_back(ctx, d, sycl::property::queue::in_order()); q2.emplace_back(ctx, d, sycl::property::queue::in_order()); }
    const size_t C = 1u << 20, PIN = 1536u << 20, DV = 512u << 20, X = 168u << 20;
    uint8_t* pin = sycl::malloc_host<uint8_t>(PIN, ctx);
    std::memset(pin, 7, PIN);
    std::vector<uint8_t*> dv(4), xr(4);
    for (int k = 1; k < 4; ++k) { dv[k] = sycl::malloc_device<uint8_t>(DV, q[k]); xr[k] = sycl::malloc_device<uint8_t>(X, q2[k]); }
    uint8_t* x0 = sycl::malloc_device<uint8_t>(X, q[0]);
    q[0].memset(x0, 1, X).wait();
    for (int k = 1; k < 4; ++k) { q[k].memcpy(dv[k], pin, C).wait(); q2[k].memcpy(xr[k], x0, C).wait(); }
    auto dma = [&](double secs) {   // pinned -> cards 1-3, 16 chunks in flight a card
        std::vector<std::vector<sycl::event>> ev(4, std::vector<sycl::event>(16));
        size_t n = 0, off = 0;
        const double t0 = now_s();
        while (now_s() - t0 < secs) {
            for (int k = 1; k < 4; ++k) {
                sycl::event& e = ev[k][(n / 3) % 16];
                e.wait();
                e = q[k].memcpy(dv[k] + (n / 3 * C) % (DV - C), pin + off, C);
                off = (off + C) % (PIN - C);
                ++n;
            }
        }
        for (int k = 1; k < 4; ++k) q[k].wait();
        return n * (double) C / (now_s() - t0) / 1e9;
    };
    for (int cpu : {0, 2, 4, 6}) {
        for (int p2p : {0, 1}) {
            std::atomic<bool> stop{false};
            std::atomic<int64_t> cbytes{0}, pbytes{0};
            std::vector<std::thread> th;
            for (int i = 0; i < cpu; ++i)
                th.emplace_back([&] {
                    std::vector<uint8_t> a(256u << 20, 1), b(256u << 20, 2);
                    while (!stop.load()) {
                        for (size_t o = 0; o + (4u << 20) <= a.size() && !stop.load(); o += 4u << 20) {
                            copy_avx(b.data() + o, a.data() + o, 4u << 20);
                            cbytes += 4 << 20;
                        }
                    }
                });
            if (p2p)
                th.emplace_back([&] {
                    while (!stop.load()) {
                        for (int k = 1; k < 4; ++k) q2[k].memcpy(xr[k], x0, X);
                        for (int k = 1; k < 4; ++k) q2[k].wait();
                        pbytes += 3 * (int64_t) X;
                    }
                });
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const int64_t c0 = cbytes.load(), p0 = pbytes.load();
            const double t0 = now_s();
            const double r = dma(T);
            const double dt = now_s() - t0;
            const double cr = (cbytes.load() - c0) / dt / 1e9, pr = (pbytes.load() - p0) / dt / 1e9;
            stop = true;
            for (auto& t : th) t.join();
            std::printf("cpu %d threads (%5.1f GB/s copied)  p2p %d (%5.1f GB/s card 0 -> 1-3): pinned -> cards 1-3 DMA %5.1f GB/s\n",
                        cpu, cr, p2p, pr, r);
            std::fflush(stdout);
        }
    }
    return 0;
}
