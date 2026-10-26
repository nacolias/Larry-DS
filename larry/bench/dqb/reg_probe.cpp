// reg_probe.cpp - larry: DMA straight from pageable memory registered for device copies
// (sycl_ext_oneapi_copy_optimize: prepare_for_device_copy / release_from_device_copy, = zeDriverImportExternalPointer)
// vs staging through a pinned buffer.  1 GiB in ~10 MB "experts", to one card and to three.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const bool thp = argc > 1 && argv[1][0] == '1';
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::context ctx(gpus);
    std::vector<sycl::queue> q;
    for (auto& d : gpus) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t total = 1u << 30, ex = 10u << 20, n = total / ex;
    void* p = nullptr;
    if (posix_memalign(&p, 2u << 20, total * 2) != 0) return 1;
    if (thp) madvise(p, total * 2, MADV_HUGEPAGE);
    uint8_t* src = (uint8_t*) p;
    std::memset(src, 3, total * 2);
    std::vector<uint8_t*> dv;
    for (auto& qq : q) dv.push_back(sycl::malloc_device<uint8_t>(total, qq));
    for (int rep = 0; rep < 3; ++rep) {
        uint8_t* s = src + (rep & 1) * total;   // alternate halves: a fresh registration each time
        // register every expert, DMA them to card 1 (or 1..3 round-robin), release
        const double t0 = now_ms();
        for (size_t i = 0; i < n; ++i) sx::prepare_for_device_copy(s + i * ex, ex, ctx);
        const double t1 = now_ms();
        for (size_t i = 0; i < n; ++i) { const int c = 1 + (int) (i % 3); q[c].memcpy(dv[c] + i * ex, s + i * ex, ex); }
        for (int c = 1; c < 4; ++c) q[c].wait();
        const double t2 = now_ms();
        for (size_t i = 0; i < n; ++i) sx::release_from_device_copy(s + i * ex, ctx);
        const double t3 = now_ms();
        std::printf("thp %d rep %d: register %.1f ms, DMA %.1f ms (%.1f GB/s), release %.1f ms -> %.1f GB/s overall\n", (int) thp, rep,
                    t1 - t0, t2 - t1, total / (t2 - t1) / 1e6, t3 - t2, total / (t3 - t0) / 1e6);
    }
    // the same bytes straight from pageable memory (driver's own path), no registration
    {
        const double t0 = now_ms();
        for (size_t i = 0; i < n; ++i) { const int c = 1 + (int) (i % 3); q[c].memcpy(dv[c] + i * ex, src + i * ex, ex); }
        for (int c = 1; c < 4; ++c) q[c].wait();
        std::printf("unregistered pageable DMA: %.1f GB/s\n", total / (now_ms() - t0) / 1e6);
    }
    return 0;
}
