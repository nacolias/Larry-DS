// membw.cpp - larry's DRAM read bandwidth with N threads (AVX2 loads, a 4 GiB buffer: no cache effects), the ceiling
// the CPU-pool expert kernels work against.  g++ -O2 -mavx2 -pthread membw.cpp -o membw
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>

int main(int argc, char ** argv) {
    const size_t bytes = (size_t) 4 << 30;
    char * buf = (char *) aligned_alloc(64, bytes);
    std::memset(buf, 1, bytes);   // fault the pages in (and make the data non-zero)
    for (int nt : { 1, 2, 4, 7, 8, 16 }) {
        std::vector<std::thread> th;
        std::vector<double> sink(nt, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nt; ++i) {
            th.emplace_back([&, i] {
                if (nt <= 8) {   // one thread a physical core (logical CPUs 0..7 on larry are the 8 cores)
                    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(i, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                }
                const size_t per = (bytes / nt) & ~(size_t) 255, b0 = per * i;
                __m256i acc = _mm256_setzero_si256();
                for (int rep = 0; rep < 2; ++rep)
                    for (size_t o = b0; o + 256 <= b0 + per; o += 256) {
                        const __m256i * p = (const __m256i *) (buf + o);
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 1));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 2));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 3));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 4));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 5));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 6));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(p + 7));
                    }
                sink[i] = (double) _mm256_extract_epi32(acc, 0);
            });
        }
        for (auto & t : th) t.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double z = 0; for (double v : sink) z += v;
        std::printf("%2d threads: %.1f GB/s (%g)\n", nt, 2.0 * bytes / s / 1e9, z * 0);
    }
    free(buf);
    return 0;
}
