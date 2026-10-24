// tools/v41/engine/xe/bench_cpu_pool.cpp - larry: the CPU expert pool on cold experts, by format, against a plain
// streaming read of the same bytes on the same threads (the DRAM's ceiling for this access pattern).  CPU only.
//
//   bench_cpu_pool [--workers 6] [--host-cpu 0] [--gib 6] [--iters 60] [--nt 1]
//
// Experts are random blobs in pageable memory (as the RAM arena), one after another through `gib` of them, so each
// run reads its expert from DRAM.  Per format (Q3_K, Q2_0, PTQ1_0 gate/up + down): a one-expert layer's time and GB/s
// through ExpertPool::run_split_multi_native (the engine's call); then the streaming read: the pool's threads (as many,
// pinned the same) summing 64-byte lines of an expert's bytes.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void pin(int c) {
    if (c < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

__attribute__((target("avx2"))) uint64_t sum_lines(const uint8_t* p, size_t n) {
    __m256i a = _mm256_setzero_si256(), b = _mm256_setzero_si256();
    for (size_t i = 0; i + 64 <= n; i += 64) {
        a = _mm256_add_epi64(a, _mm256_loadu_si256((const __m256i*) (p + i)));
        b = _mm256_add_epi64(b, _mm256_loadu_si256((const __m256i*) (p + i + 32)));
    }
    a = _mm256_add_epi64(a, b);
    alignas(32) uint64_t t[4];
    _mm256_store_si256((__m256i*) t, a);
    return t[0] + t[1] + t[2] + t[3];
}

}  // namespace

int main(int argc, char** argv) {
    int workers = 6, host_cpu = 0, iters = 60, nt = 1;
    double gib = 6.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--workers") workers = std::atoi(argv[i + 1]);
        else if (k == "--host-cpu") host_cpu = std::atoi(argv[i + 1]);
        else if (k == "--gib") gib = std::atof(argv[i + 1]);
        else if (k == "--iters") iters = std::atoi(argv[i + 1]);
        else if (k == "--nt") nt = std::atoi(argv[i + 1]);
    }
    // the pool first: its topology comes from the calling thread's mask (pinned first, it saw CPU 0 alone)
    const std::vector<int> all_cores = cpu::physical_cores(true);
    cpu::ExpertPool pool(workers);
    pin(host_cpu);
    std::printf("pool: %d workers (+ host on CPU %d), host works %d\n", pool.workers(), host_cpu, (int) pool.host_works());
    const int64_t H = 5120, F = 2304;
    struct T { int gu, d; const char* name; };
    const T types[] = { { 11, 11, "Q3_K" }, { 42, 42, "Q2_0" }, { 143, 143, "PTQ1_0" } };
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) H * 8);
    for (auto& v : x) v = nd(rng);
    for (const T& t : types) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(t.gu, t.d, H, F, f, err)) { std::printf("%s: %s\n", t.name, err.c_str()); continue; }
        f.swiglu_limit = 10.0f;
        const size_t nexp = std::max<size_t>(2, (size_t) (gib * 1073741824.0 / (double) f.bytes));
        std::vector<uint8_t> arena(nexp * f.bytes);
        for (size_t i = 0; i < arena.size(); i += 4096) arena[i] = (uint8_t) rng();   // touched (faulted in)
        {   // random contents (the decoders do not care, but zero pages could alias)
            uint32_t s = 12345;
            for (size_t i = 0; i < arena.size(); i += 4) { s = s * 1664525u + 1013904223u; std::memcpy(&arena[i], &s, std::min<size_t>(4, arena.size() - i)); }
        }
        // activations, as cpu_ents makes them
        std::vector<cpu::ActQ> aq((size_t) nt);
        std::vector<std::vector<uint8_t>> na((size_t) nt, std::vector<uint8_t>(cpu::kNativeActBytes));
        const bool q2 = cpu::q2_native_kernels(f.gu_type);
        for (int s = 0; s < nt; ++s) {
            if (q2) cpu::act_quant_any(x.data() + (size_t) s * H, (int) H, aq[(size_t) s]);
            else cpu::native_quant_act(f, x.data() + (size_t) s * H, na[(size_t) s].data());
        }
        std::vector<std::vector<float>> out((size_t) nt, std::vector<float>((size_t) H));
        cpu::ExpertJobMulti j;
        j.fmt = &f;
        j.nt = nt;
        for (int s = 0; s < nt; ++s) { j.act[s] = &aq[(size_t) s]; j.nact[s] = q2 ? nullptr : na[(size_t) s].data(); j.out[s] = out[(size_t) s].data(); }
        double best = 1e30, sum = 0;
        int n = 0;
        for (int it = 0; it < iters; ++it) {
            j.blob = arena.data() + (size_t) (it % (int) nexp) * f.bytes;
            const double t0 = now_us();
            pool.run_split_multi_native(f, &j, 1);
            const double us = now_us() - t0;
            if (it >= 4) { sum += us; ++n; best = std::min(best, us); }
        }
        const double avg = sum / std::max(1, n);
        // the streaming read: the same threads, an expert's bytes split in equal parts (after the pool's workers went to
        // sleep: they spin on these CPUs for 20 ms after a run)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int th = pool.workers() + 1;
        std::vector<int> cores;
        for (int c : all_cores) if ((int) cores.size() < pool.workers()) cores.push_back(c);
        while ((int) cores.size() < pool.workers()) cores.push_back(-1);
        double rsum = 0;
        int rn = 0;
        std::atomic<uint64_t> sink{0};
        for (int it = 0; it < std::min(iters, 30); ++it) {
            const uint8_t* b = arena.data() + (size_t) ((it * 7 + 3) % (int) nexp) * f.bytes;
            const size_t part = (f.bytes / (size_t) th) & ~(size_t) 63;
            std::vector<std::thread> ts;
            std::atomic<int> ready{0}, go{0};
            std::vector<double> t_end((size_t) th);
            for (int k = 1; k < th; ++k)
                ts.emplace_back([&, k] {
                    pin(cores[(size_t) k - 1]);
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) _mm_pause();
                    sink += sum_lines(b + (size_t) k * part, part);
                });
            while (ready.load() != th - 1) _mm_pause();
            const double t0 = now_us();
            go.store(1, std::memory_order_release);
            sink += sum_lines(b, part);
            for (auto& tt : ts) tt.join();
            const double us = now_us() - t0;
            if (it >= 2) { rsum += us; ++rn; }
        }
        const double ravg = rsum / std::max(1, rn);
        std::printf("%-6s %5.2f MB a blob, %zu blobs: pool %6.0f us a layer (best %5.0f) = %5.1f GB/s | streaming read %5.0f us = %5.1f GB/s  (nt %d)\n",
                    t.name, f.bytes / 1e6, nexp, avg, best, f.bytes / avg / 1e3, ravg, f.bytes / ravg / 1e3, nt);
        (void) sink.load();
    }
    return 0;
}
