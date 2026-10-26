// uptr_probe.cpp - larry: can the RAM arena's pageable chunks (anonymous memory, MADV_HUGEPAGE) be DMA'd straight onto
// the cards once imported (SYCL prepare_for_device_copy -> zexDriverImportExternalPointer: an xe userptr) instead of
// through the hot rings (a CPU copy into pinned ring lines + the DMA: ~3 DRAM bytes a byte)?  Cards 1-3 copy 1 MiB
// chunks, 16 in flight a card, for T s from
//   pinned   malloc_host (what the pinned arena is)
//   import   anonymous THP memory after prepare_for_device_copy
//   plain    the same kind of memory without the import (the runtime's own staging)
// each alone and with N CPU threads streaming memory (the CPU pool's DRAM load); import's bytes checked on card 1 and
// a run with a forced compaction every 0.5 s (do invalidations show up?).  Prints GB/s, the import's time and the
// process's AnonHugePages.  Run with the service stopped: uptr_probe [GiB a source] [s]
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__attribute__((target("avx2"))) static void copy_avx(uint8_t* d, const uint8_t* s, size_t n) {
    for (size_t i = 0; i + 128 <= n; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i*) (s + i)), b = _mm256_loadu_si256((const __m256i*) (s + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i*) (s + i + 64)), e = _mm256_loadu_si256((const __m256i*) (s + i + 96));
        _mm256_storeu_si256((__m256i*) (d + i), a); _mm256_storeu_si256((__m256i*) (d + i + 32), b);
        _mm256_storeu_si256((__m256i*) (d + i + 64), c); _mm256_storeu_si256((__m256i*) (d + i + 96), e);
    }
}

static long kb_of(const char* file, const char* key) {   // a "Key:   N kB" line
    std::ifstream f(file);
    std::string k;
    long v = -1;
    while (f >> k) {
        if (k == key) { f >> v; return v; }
        std::getline(f, k);
    }
    return -1;
}

static uint8_t* anon(size_t n) {   // 2 MiB aligned anonymous memory, huge pages asked for, written once
    const size_t A = 2u << 20;
    void* p = mmap(nullptr, n + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    uint8_t* q = (uint8_t*) (((uintptr_t) p + A - 1) & ~(uintptr_t) (A - 1));
    madvise(q, n, MADV_HUGEPAGE);
    for (size_t i = 0; i < n; i += 4096) q[i] = (uint8_t) (i >> 12);
    for (size_t i = 0; i < n; i += 64) q[i + 17] = (uint8_t) (i * 2654435761u >> 24);
    return q;
}

int main(int argc, char** argv) {
    const size_t G = (size_t) (argc > 1 ? std::atof(argv[1]) : 2.0) * (1u << 30);
    const double T = argc > 2 ? std::atof(argv[2]) : 2.0;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 4) { std::fprintf(stderr, "need 4 cards\n"); return 1; }
    sycl::context ctx(devs);
    std::vector<sycl::queue> q;
    for (auto& d : devs) q.emplace_back(ctx, d, sycl::property::queue::in_order());
    const size_t C = 1u << 20, DV = 512u << 20;
    std::vector<uint8_t*> dv(4);
    for (int k = 1; k < 4; ++k) dv[k] = sycl::malloc_device<uint8_t>(DV, q[k]);
    const long ahp0 = kb_of("/proc/self/smaps_rollup", "AnonHugePages:");
    uint8_t* pin = sycl::malloc_host<uint8_t>(G, ctx);
    std::memset(pin, 7, G);
    uint8_t* imp = anon(G);
    uint8_t* pla = anon(G);
    if (!pin || !imp || !pla) { std::fprintf(stderr, "allocation failed\n"); return 1; }
    const long ahp1 = kb_of("/proc/self/smaps_rollup", "AnonHugePages:");
    double t0 = now_s();
    sx::prepare_for_device_copy(imp, G, ctx);
    const double t_imp = now_s() - t0;
    std::printf("sources %.1f GiB each; AnonHugePages +%.2f GiB of the two anonymous ones; import %.1f ms\n",
                G / 1073741824.0, (ahp1 - ahp0) / 1048576.0, t_imp * 1e3);
    for (int k = 1; k < 4; ++k) {   // first touches (residency) outside the timing
        q[k].memcpy(dv[k], pin, C).wait();
        q[k].memcpy(dv[k], imp, C).wait();
        q[k].memcpy(dv[k], pla, C).wait();
    }
    // correctness: 64 MiB of the imported memory onto card 1 and back
    {
        const size_t N = 64u << 20, off = G / 3 & ~(size_t) 4095;
        std::vector<uint8_t> back(N, 0);
        q[1].memcpy(dv[1], imp + off, N).wait();
        q[1].memcpy(back.data(), dv[1], N).wait();
        std::printf("import: 64 MiB round trip %s\n", std::memcmp(back.data(), imp + off, N) == 0 ? "matches" : "DIFFERS");
    }
    auto dma = [&](const uint8_t* src, double secs) {   // src -> cards 1-3, 16 chunks in flight a card
        std::vector<std::vector<sycl::event>> ev(4, std::vector<sycl::event>(16));
        size_t n = 0, off = 0;
        const double s0 = now_s();
        while (now_s() - s0 < secs) {
            for (int k = 1; k < 4; ++k) {
                sycl::event& e = ev[k][(n / 3) % 16];
                e.wait();
                e = q[k].memcpy(dv[k] + (n / 3 * C) % (DV - C), src + off, C);
                off = (off + C) % (G - C);
                ++n;
            }
        }
        for (int k = 1; k < 4; ++k) q[k].wait();
        return n * (double) C / (now_s() - s0) / 1e9;
    };
    const char* name[3] = {"pinned", "import", "plain"};
    const uint8_t* src[3] = {pin, imp, pla};
    for (int cpu : {0, 2, 4}) {
        for (int s = 0; s < 3; ++s) {
            std::atomic<bool> stop{false};
            std::atomic<int64_t> cbytes{0};
            std::vector<std::thread> th;
            for (int i = 0; i < cpu; ++i)
                th.emplace_back([&] {
                    std::vector<uint8_t> a(256u << 20, 1), b(256u << 20, 2);
                    while (!stop.load())
                        for (size_t o = 0; o + (4u << 20) <= a.size() && !stop.load(); o += 4u << 20) {
                            copy_avx(b.data() + o, a.data() + o, 4u << 20);
                            cbytes += 4 << 20;
                        }
                });
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const int64_t c0 = cbytes.load();
            const double s0 = now_s();
            const double r = dma(src[s], T);
            const double cr = (cbytes.load() - c0) / (now_s() - s0) / 1e9;
            stop = true;
            for (auto& t : th) t.join();
            std::printf("cpu %d threads (%5.1f GB/s copied): %-6s -> cards 1-3 %5.1f GB/s\n", cpu, cr, name[s], r);
            std::fflush(stdout);
        }
    }
    // the imported memory while the kernel compacts every 0.5 s (page migrations: userptr invalidations?)
    {
        std::atomic<bool> stop{false};
        std::thread cm([&] {
            while (!stop.load()) {
                if (FILE* f = std::fopen("/proc/sys/vm/compact_memory", "w")) { std::fputs("1", f); std::fclose(f); }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
        const long m0 = kb_of("/proc/vmstat", "pgmigrate_success");
        const double r = dma(imp, 3 * T);
        stop = true;
        cm.join();
        std::printf("import while compacting: %5.1f GB/s (pgmigrate_success +%ld)\n", r, kb_of("/proc/vmstat", "pgmigrate_success") - m0);
    }
    sx::release_from_device_copy(imp, ctx);
    for (int k = 1; k < 4; ++k) sycl::free(dv[k], ctx);
    sycl::free(pin, ctx);
    return 0;
}
