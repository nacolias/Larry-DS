// h2d_bw.cpp - larry: host -> card copy bandwidth from pinned memory, one card at a time and several at once (each
// card's own in-order queue, 64 MiB chunks, 1 GiB per card), plus pageable -> pinned memcpy rates (glibc memcpy vs
// non-temporal AVX2 stores) for the prompt prefetch's staging.
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void nt_copy(void* dst, const void* src, size_t n) {   // 32-byte aligned dst, n % 128 == 0
    auto* d = (__m256i*) dst;
    auto* s = (const __m256i*) src;
    for (size_t i = 0; i < n / 32; i += 4) {
        __m256i a = _mm256_loadu_si256(s + i), b = _mm256_loadu_si256(s + i + 1);
        __m256i c = _mm256_loadu_si256(s + i + 2), e = _mm256_loadu_si256(s + i + 3);
        _mm256_stream_si256(d + i, a); _mm256_stream_si256(d + i + 1, b);
        _mm256_stream_si256(d + i + 2, c); _mm256_stream_si256(d + i + 3, e);
    }
    _mm_sfence();
}

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto& d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t chunk = 64u << 20, per = 1u << 30;
    uint8_t* h = sycl::malloc_host<uint8_t>(per, ctx);
    std::memset(h, 1, per);
    std::vector<uint8_t*> dv;
    for (auto& qq : q) dv.push_back(sycl::malloc_device<uint8_t>(per, qq));
    auto run = [&](std::vector<int> cards) {
        for (int c : cards) q[c].memcpy(dv[c], h, chunk).wait();   // warm
        const double t0 = now_s();
        for (size_t off = 0; off < per; off += chunk)
            for (int c : cards) q[c].memcpy(dv[c] + off, h + off, chunk);
        for (int c : cards) q[c].wait();
        const double dt = now_s() - t0;
        std::printf("cards");
        for (int c : cards) std::printf(" %d", c);
        std::printf(": %.1f GB/s total (%.2f GB in %.3f s)\n", cards.size() * per / dt / 1e9, cards.size() * per / 1e9, dt);
    };
    run({0}); run({1}); run({2}); run({3});
    run({1, 2}); run({1, 3}); run({2, 3}); run({1, 2, 3}); run({0, 1, 2, 3});
    // staging: pageable -> pinned, 1..6 threads, glibc memcpy vs non-temporal
    const size_t sb = 512u << 20;
    std::vector<uint8_t*> src;
    for (int t = 0; t < 8; ++t) { src.push_back((uint8_t*) aligned_alloc(64, sb)); std::memset(src.back(), 2, sb); }
    uint8_t* pin = sycl::malloc_host<uint8_t>(8 * sb, ctx);
    for (int mode = 0; mode < 2; ++mode)
        for (int nt : {1, 2, 4, 6, 8}) {
            std::vector<std::thread> th;
            const double t0 = now_s();
            for (int t = 0; t < nt; ++t)
                th.emplace_back([&, t] {
                    for (size_t off = 0; off < sb; off += 16u << 20) {
                        if (mode == 0) std::memcpy(pin + t * sb + off, src[t] + off, 16u << 20);
                        else nt_copy(pin + t * sb + off, src[t] + off, 16u << 20);
                    }
                });
            for (auto& x : th) x.join();
            const double dt = now_s() - t0;
            std::printf("staging %s, %d threads: %.1f GB/s (%.1f a thread)\n", mode ? "non-temporal" : "memcpy", nt, nt * sb / dt / 1e9,
                        sb / dt / 1e9);
        }
    return 0;
}
