// stage_ring.cpp - larry: the prompt path's pageable expert copies end to end (pageable host memory -> pinned bounce ->
// the expert cards 1-3), the tier's way and a ring:
//   stage  the prefetch as the tier does it: NT stager threads memcpy whole experts (PIECE bytes) into a big pinned
//          stage (two slots of a layer each, alternating like pf[0] / pf[1]); then the layer's DMAs go on the cards'
//          queues (an expert a copy, card by card) while the stagers fill the other slot
//   ring   a small pinned ring of R chunks of C bytes: the stagers fill chunks in order, one submit thread puts each
//          filled chunk's DMA on the next card's queue at once and recycles the chunk when that copy is done.  A ring
//          that fits the L3 (32 MB) may keep the staged lines in the caches until the card reads them (if its reads
//          snoop): one DRAM trip a byte (the source read) instead of three (read, write, the DMA's read)
// plus the PCIe ceiling (DMA straight from pinned memory) and the CPU-only copy rates.  Only the submit thread calls
// SYCL (the tier's rule: the Level Zero driver must not see several submitting threads while pageable copies run).
// Run with the service stopped: stage_ring [GiB a test, default 6]
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void pin_cpu(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

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

static void tmp_copy(void* dst, const void* src, size_t n) {   // plain AVX2 stores (the lines stay in the caches)
    auto* d = (__m256i*) dst;
    auto* s = (const __m256i*) src;
    for (size_t i = 0; i < n / 32; i += 4) {
        __m256i a = _mm256_loadu_si256(s + i), b = _mm256_loadu_si256(s + i + 1);
        __m256i c = _mm256_loadu_si256(s + i + 2), e = _mm256_loadu_si256(s + i + 3);
        _mm256_store_si256(d + i, a); _mm256_store_si256(d + i + 1, b);
        _mm256_store_si256(d + i + 2, c); _mm256_store_si256(d + i + 3, e);
    }
}

enum CopyKind { K_LIBC = 0, K_AVX = 1, K_NT = 2 };
static const char* kname[] = {"memcpy", "avx", "nt"};

static void copy_k(int k, void* d, const void* s, size_t n) {
    if (k == K_LIBC) std::memcpy(d, s, n);
    else if (k == K_AVX) tmp_copy(d, s, n);
    else nt_copy(d, s, n);
}

struct Ctx {
    sycl::context ctx;
    std::vector<sycl::queue> q;
    std::vector<int> cards;          // the expert cards
    std::vector<uint8_t*> dv;        // a device buffer a card
    size_t dcap = 0;
    uint8_t* src = nullptr;          // pageable "arena"
    size_t total = 0;
};

// the PCIe ceiling: a pinned buffer straight onto the cards, pieces round-robin
static void t_pcie(Ctx& c, uint8_t* pin, size_t bytes, size_t piece) {
    const double t0 = now_s();
    size_t i = 0;
    for (size_t off = 0; off + piece <= bytes; off += piece, ++i) {
        const int k = c.cards[i % c.cards.size()];
        c.q[k].memcpy(c.dv[k] + (i / c.cards.size() * piece) % (c.dcap - piece), pin + off, piece);
    }
    for (int k : c.cards) c.q[k].wait();
    const double dt = now_s() - t0;
    std::printf("pcie   pinned -> %zu cards, %5.1f MiB pieces: %5.1f GB/s\n", c.cards.size(), piece / 1048576.0, bytes / dt / 1e9);
}

// CPU only: nt threads copy the source into a destination of `dcap` bytes (cycled)
static void t_cpu(Ctx& c, uint8_t* dst, size_t dcap, size_t chunk, int nt, int kind) {
    const size_t n = c.total / chunk;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            pin_cpu(1 + t);
            for (size_t i; (i = next.fetch_add(1)) < n;)
                copy_k(kind, dst + (i * chunk) % (dcap - chunk + 1) / 256 * 256, c.src + i * chunk, chunk);
        });
    for (auto& x : th) x.join();
    const double dt = now_s() - t0;
    std::printf("cpu    %-6s %d threads into %7.1f MiB, %4.1f MiB chunks: %5.1f GB/s\n", kname[kind], nt, dcap / 1048576.0,
                chunk / 1048576.0, n * chunk / dt / 1e9);
}

// the tier's prefetch: two layer slots of `slot` bytes; stagers fill one while the other's DMAs run
static void t_stage(Ctx& c, uint8_t* stage, size_t slot, size_t piece, int nt, int kind) {
    const size_t per_layer = slot / piece;
    const size_t layers = c.total / (per_layer * piece);
    std::vector<std::vector<sycl::event>> ev(2);
    const double t0 = now_s();
    size_t di = 0;
    for (size_t L = 0; L < layers; ++L) {
        const int s = (int) (L & 1);
        for (auto& e : ev[s]) e.wait();   // the slot's previous layer is on the cards
        ev[s].clear();
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                pin_cpu(1 + t);
                for (size_t i; (i = next.fetch_add(1)) < per_layer;)
                    copy_k(kind, stage + s * slot + i * piece, c.src + (L * per_layer + i) * piece, piece);
            });
        for (auto& x : th) x.join();
        for (size_t i = 0; i < per_layer; ++i, ++di) {
            const int k = c.cards[di % c.cards.size()];
            ev[s].push_back(c.q[k].memcpy(c.dv[k] + (di / c.cards.size() * piece) % (c.dcap - piece), stage + s * slot + i * piece, piece));
        }
    }
    for (int k : c.cards) c.q[k].wait();
    const double dt = now_s() - t0;
    std::printf("stage  %-6s %d stagers, slots 2 x %6.0f MiB, %5.1f MiB pieces: %5.1f GB/s\n", kname[kind], nt, slot / 1048576.0,
                piece / 1048576.0, layers * per_layer * piece / dt / 1e9);
}

// the ring: R chunks of C bytes; piece i uses chunk i % R once the DMA of piece i - R is done
static void t_ring(Ctx& c, uint8_t* ring, size_t R, size_t C, int nt, int kind) {
    const size_t n = c.total / C;
    std::vector<std::atomic<int64_t>> ready(R);
    for (auto& r : ready) r.store(-1);
    std::atomic<int64_t> done_upto{0};   // every piece below this is on its card
    std::atomic<size_t> next{0};
    std::atomic<bool> quit{false};
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            pin_cpu(1 + t);
            for (size_t i; (i = next.fetch_add(1)) < n;) {
                while ((int64_t) i - (int64_t) R >= done_upto.load(std::memory_order_acquire)) {
                    if (quit.load()) return;
                    _mm_pause();
                }
                copy_k(kind, ring + (i % R) * C, c.src + i * C, C);
                ready[i % R].store((int64_t) i, std::memory_order_release);
            }
        });
    pin_cpu(0);
    std::deque<std::pair<int64_t, sycl::event>> fl;
    auto poll = [&](bool block) {
        while (!fl.empty()) {
            auto& e = fl.front().second;
            if (block) e.wait();
            else if (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) return;
            done_upto.store(fl.front().first + 1, std::memory_order_release);
            fl.pop_front();
            block = false;
        }
    };
    double t_wait = 0;
    for (size_t i = 0; i < n; ++i) {
        const double tw = now_s();
        while (ready[i % R].load(std::memory_order_acquire) != (int64_t) i) {
            poll(false);
            _mm_pause();
        }
        t_wait += now_s() - tw;
        const int k = c.cards[i % c.cards.size()];
        fl.emplace_back((int64_t) i, c.q[k].memcpy(c.dv[k] + (i / c.cards.size() * C) % (c.dcap - C), ring + (i % R) * C, C));
        if (fl.size() >= R) poll(true);
        else poll(false);
    }
    while (!fl.empty()) poll(true);
    quit = true;
    for (auto& x : th) x.join();
    const double dt = now_s() - t0;
    std::printf("ring   %-6s %d stagers, %3zu x %4.2f MiB = %6.1f MiB: %5.1f GB/s (submitter waited %.0f %% for chunks)\n",
                kname[kind], nt, R, C / 1048576.0, R * C / 1048576.0, n * C / dt / 1e9, 100.0 * t_wait / dt);
}

int main(int argc, char** argv) {
    const double gib = argc > 1 ? std::atof(argv[1]) : 6.0;
    Ctx c;
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 4) { std::fprintf(stderr, "need 4 cards (%zu)\n", gpus.size()); return 1; }
    c.ctx = sycl::context(gpus);
    for (auto& d : gpus) c.q.emplace_back(c.ctx, d, sycl::property::queue::in_order());
    c.cards = {1, 2, 3};
    c.dcap = 1ull << 30;
    c.dv.assign(gpus.size(), nullptr);
    for (int k : c.cards) c.dv[k] = sycl::malloc_device<uint8_t>(c.dcap, c.q[k]);
    c.total = (size_t) (gib * 1073741824.0) & ~((size_t) (16 << 20) - 1);
    c.src = (uint8_t*) std::aligned_alloc(4096, c.total);
    {   // fault the source in with data that does not compress
        uint64_t x = 0x9e3779b97f4a7c15ull;
        auto* p = (uint64_t*) c.src;
        for (size_t i = 0; i < c.total / 8; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; p[i] = x; }
    }
    const size_t piece = 11141120;              // ~10.6 MiB: an expert (79.4 GiB / 7,667)
    const size_t slot = 144 * piece;            // a layer's pageable experts (~1.5 GiB)
    uint8_t* stage = sycl::malloc_host<uint8_t>(2 * slot, c.ctx);
    uint8_t* ring = sycl::malloc_host<uint8_t>(256u << 20, c.ctx);
    if (!stage || !ring) { std::fprintf(stderr, "pinned allocation failed\n"); return 1; }
    std::memset(stage, 1, 2 * slot);
    std::memset(ring, 1, 256u << 20);
    for (int k : c.cards) {   // warm the copies (first P2P / H2D per queue)
        c.q[k].memcpy(c.dv[k], stage, 64u << 20).wait();
        c.q[k].memcpy(c.dv[k], ring, 1u << 20).wait();
    }
    std::printf("%.2f GiB a test, cards 1-3, piece %.1f MiB\n", c.total / 1073741824.0, piece / 1048576.0);
    std::memcpy(stage, c.src, 2 * slot);
    t_pcie(c, stage, 2 * slot, piece);
    t_pcie(c, stage, 2 * slot, piece);
    t_pcie(c, ring, 256u << 20, 1u << 20);
    for (int kind : {K_LIBC, K_AVX, K_NT}) {
        t_cpu(c, stage, 2 * slot, piece, 4, kind);
        t_cpu(c, ring, 16u << 20, 1u << 20, 4, kind);
    }
    for (int rep = 0; rep < 2; ++rep)
        for (int kind : {K_LIBC, K_NT}) t_stage(c, stage, slot, piece, 4, kind);
    for (int rep = 0; rep < 2; ++rep)
        for (int kind : {K_AVX, K_NT})
            for (size_t C : {1u << 20, 2u << 20, 4u << 20})
                for (size_t mib : {8, 16, 32, 64, 256}) {
                    const size_t R = (mib << 20) / C;
                    if (R < 4) continue;
                    t_ring(c, ring, R, C, 4, kind);
                }
    for (int nt : {2, 3, 6}) t_ring(c, ring, 16, 1u << 20, nt, K_AVX);
    for (int nt : {2, 3, 6}) t_ring(c, ring, 8, 2u << 20, nt, K_AVX);
    for (int nt : {2, 6}) t_stage(c, stage, slot, piece, nt, K_LIBC);
    for (int k : c.cards) sycl::free(c.dv[k], c.q[k]);
    sycl::free(stage, c.ctx);
    sycl::free(ring, c.ctx);
    std::free(c.src);
    return 0;
}
