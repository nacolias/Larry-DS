// bar_probe.cpp - larry: can the prefetch stagers write experts straight into a card's VRAM through its 32 GiB BAR?
// The CPU's read of the pageable arena would be the only DRAM byte (no ring line written back, no DMA read of it) and
// the card's one copy engine would stay free for the pinned arena's DMAs.  On card 1 (0000:1b:00.0): a VRAM buffer
// object made through the xe uAPI (NEEDS_VISIBLE_VRAM, WC) and mmap'd; the CPU's write GB/s with 1 / 2 / 4 threads
// (streaming and plain stores) alone and while pinned host memory DMAs onto cards 1-3; the PRIME export's time and the
// host RAM it takes (xe may populate a system copy of the buffer on export); the buffer imported into Level Zero as a
// device allocation of card 1 and its bytes read back by the card and compared.  Run with the service stopped:
//   bar_probe [MiB, default 1024]
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <drm/drm.h>
#include <drm/xe_drm.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static long avail_kb() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    long v;
    while (f >> k >> v) {
        if (k == "MemAvailable:") return v;
        std::getline(f, k);
    }
    return -1;
}

__attribute__((target("avx2"))) static void put_nt(uint8_t* d, const uint8_t* s, size_t n) {   // streaming stores
    for (size_t i = 0; i + 128 <= n; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i*) (s + i)), b = _mm256_loadu_si256((const __m256i*) (s + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i*) (s + i + 64)), e = _mm256_loadu_si256((const __m256i*) (s + i + 96));
        _mm256_stream_si256((__m256i*) (d + i), a); _mm256_stream_si256((__m256i*) (d + i + 32), b);
        _mm256_stream_si256((__m256i*) (d + i + 64), c); _mm256_stream_si256((__m256i*) (d + i + 96), e);
    }
    _mm_sfence();
}

__attribute__((target("avx2"))) static void put_plain(uint8_t* d, const uint8_t* s, size_t n) {
    for (size_t i = 0; i + 128 <= n; i += 128) {
        __m256i a = _mm256_loadu_si256((const __m256i*) (s + i)), b = _mm256_loadu_si256((const __m256i*) (s + i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i*) (s + i + 64)), e = _mm256_loadu_si256((const __m256i*) (s + i + 96));
        _mm256_storeu_si256((__m256i*) (d + i), a); _mm256_storeu_si256((__m256i*) (d + i + 32), b);
        _mm256_storeu_si256((__m256i*) (d + i + 64), c); _mm256_storeu_si256((__m256i*) (d + i + 96), e);
    }
    _mm_sfence();
}

static std::string render_node(const char* pci) {   // /dev/dri/renderD* of a PCI address
    std::string dir = std::string("/sys/bus/pci/devices/") + pci + "/drm";
    DIR* d = opendir(dir.c_str());
    if (!d) return "";
    std::string r;
    while (dirent* e = readdir(d))
        if (!std::strncmp(e->d_name, "renderD", 7)) r = std::string("/dev/dri/") + e->d_name;
    closedir(d);
    return r;
}

int main(int argc, char** argv) {
    const size_t S = (size_t) (argc > 1 ? std::atol(argv[1]) : 1024) << 20;
    const char* PCI = "0000:1b:00.0";
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 4) { std::fprintf(stderr, "need 4 cards\n"); return 1; }
    sycl::context ctx(devs);
    std::vector<sycl::queue> q;
    int k1 = -1;
    for (size_t i = 0; i < devs.size(); ++i) {
        q.emplace_back(ctx, devs[i], sycl::property::queue::in_order());
        if (devs[i].get_info<sycl::ext::intel::info::device::pci_address>() == PCI) k1 = (int) i;
    }
    if (k1 < 0) { std::fprintf(stderr, "no SYCL device at %s\n", PCI); return 1; }
    const std::string node = render_node(PCI);
    const int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) { std::perror(node.c_str()); return 1; }
    // the VRAM region
    drm_xe_device_query dq = {};
    dq.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
    if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq) || !dq.size) { std::perror("query size"); return 1; }
    std::vector<uint8_t> qb(dq.size);
    dq.data = (uintptr_t) qb.data();
    if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq)) { std::perror("query"); return 1; }
    auto* mr = (drm_xe_query_mem_regions*) qb.data();
    int vinst = -1;
    for (uint32_t i = 0; i < mr->num_mem_regions; ++i) {
        const drm_xe_mem_region& r = mr->mem_regions[i];
        std::printf("region %u: class %u instance %u min page %u total %.2f GiB cpu-visible %.2f GiB\n", i, r.mem_class, r.instance,
                    r.min_page_size, r.total_size / 1073741824.0, r.cpu_visible_size / 1073741824.0);
        if (r.mem_class == DRM_XE_MEM_REGION_CLASS_VRAM && vinst < 0) vinst = r.instance;
    }
    if (vinst < 0) { std::fprintf(stderr, "no VRAM region\n"); return 1; }
    drm_xe_gem_create gc = {};
    gc.size = S;
    gc.placement = 1u << vinst;
    gc.flags = DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM;
    gc.cpu_caching = DRM_XE_GEM_CPU_CACHING_WC;
    if (ioctl(fd, DRM_IOCTL_XE_GEM_CREATE, &gc)) { std::perror("gem create"); return 1; }
    drm_xe_gem_mmap_offset mo = {};
    mo.handle = gc.handle;
    if (ioctl(fd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mo)) { std::perror("mmap offset"); return 1; }
    uint8_t* bar = (uint8_t*) mmap(nullptr, S, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mo.offset);
    if (bar == MAP_FAILED) { std::perror("mmap"); return 1; }
    // a pageable source like the arena's, written once
    std::vector<uint8_t> src(S);
    for (size_t i = 0; i < S; i += 8) { uint64_t v = i * 0x9E3779B97F4A7C15ull; std::memcpy(&src[i], &v, 8); }
    auto put_mt = [&](int nt, bool nts) {
        std::vector<std::thread> th;
        const size_t part = S / nt / 128 * 128;
        const double t0 = now_s();
        for (int t = 0; t < nt; ++t)
            th.emplace_back([&, t] { (nts ? put_nt : put_plain)(bar + t * part, src.data() + t * part, part); });
        for (auto& x : th) x.join();
        return (double) part * nt / (now_s() - t0) / 1e9;
    };
    put_mt(4, true);   // first touch (the BO's backing)
    for (int nts : {1, 0})
        for (int nt : {1, 2, 4, 6}) {
            double best = 0;
            for (int r = 0; r < 3; ++r) best = std::max(best, put_mt(nt, nts));
            std::printf("CPU -> VRAM through the BAR: %d threads, %s stores: %5.1f GB/s\n", nt, nts ? "streaming" : "plain", best);
        }
    // the PRIME export (a populated system copy?) and the Level Zero import
    long a0 = avail_kb();
    double t0 = now_s();
    drm_prime_handle ph = {};
    ph.handle = gc.handle;
    ph.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph)) { std::perror("prime export"); return 1; }
    const double t_exp = now_s() - t0;
    long a1 = avail_kb();
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devs[(size_t) k1]);
    ze_external_memory_import_fd_t imp = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr, ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, ph.fd};
    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &imp, 0, 0};
    void* dp = nullptr;
    t0 = now_s();
    ze_result_t zr = zeMemAllocDevice(zc, &dd, S, 4096, zd, &dp);
    const double t_imp = now_s() - t0;
    long a2 = avail_kb();
    std::printf("PRIME export %.1f ms (MemAvailable %+.2f GiB), Level Zero import %s %.1f ms (%+.2f GiB)\n", t_exp * 1e3,
                (a1 - a0) / 1048576.0, zr == ZE_RESULT_SUCCESS ? "ok" : "FAILED", t_imp * 1e3, (a2 - a1) / 1048576.0);
    if (zr != ZE_RESULT_SUCCESS) return 1;
    // the card reads what the CPU wrote (a fresh pattern, streaming stores), through its import
    for (size_t i = 0; i < S; i += 8) { uint64_t v = ~(i * 0x9E3779B97F4A7C15ull); std::memcpy(&src[i], &v, 8); }
    put_mt(4, true);
    volatile uint8_t sink = bar[S - 1];   // a read through the BAR: the posted writes are out
    (void) sink;
    uint8_t* dv = sycl::malloc_device<uint8_t>(S, q[(size_t) k1]);
    std::vector<uint8_t> back(S, 0);
    q[(size_t) k1].memcpy(dv, dp, S).wait();
    q[(size_t) k1].memcpy(back.data(), dv, S).wait();
    std::printf("card 1 reads the imported buffer: %s\n", std::memcmp(back.data(), src.data(), S) == 0 ? "matches" : "DIFFERS");
    // with pinned DMAs onto cards 1-3 at the same time
    {
        const size_t PIN = 1024u << 20, C = 1u << 20, DV = 256u << 20;
        uint8_t* pin = sycl::malloc_host<uint8_t>(PIN, ctx);
        std::memset(pin, 3, PIN);
        std::vector<uint8_t*> dd2(devs.size(), nullptr);
        for (size_t k = 0; k < devs.size(); ++k) if ((int) k != 0) dd2[k] = sycl::malloc_device<uint8_t>(DV, q[k]);
        std::atomic<bool> stop{false};
        std::atomic<int64_t> dbytes{0};
        std::thread dma([&] {
            std::vector<std::vector<sycl::event>> ev(devs.size(), std::vector<sycl::event>(16));
            size_t n = 0, off = 0;
            while (!stop.load()) {
                for (size_t k = 1; k < devs.size(); ++k) {
                    sycl::event& e = ev[k][n % 16];
                    e.wait();
                    e = q[k].memcpy(dd2[k] + (n * C) % (DV - C), pin + off, C);
                    off = (off + C) % (PIN - C);
                    dbytes += C;
                }
                ++n;
            }
            for (size_t k = 1; k < devs.size(); ++k) q[k].wait();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        for (int nt : {2, 4}) {
            const int64_t b0 = dbytes.load();
            const double s0 = now_s();
            double w = 0;
            int reps = 0;
            while (now_s() - s0 < 2.0) { w += put_mt(nt, true); ++reps; }
            std::printf("with pinned DMAs onto cards 1-3: CPU -> VRAM %d threads %5.1f GB/s, DMA %5.1f GB/s\n", nt, w / reps,
                        (dbytes.load() - b0) / (now_s() - s0) / 1e9);
        }
        stop = true;
        dma.join();
        for (size_t k = 1; k < devs.size(); ++k) sycl::free(dd2[k], ctx);
        sycl::free(pin, ctx);
    }
    sycl::free(dv, ctx);
    zeMemFree(zc, dp);
    munmap(bar, S);
    close(ph.fd);
    drm_gem_close gcl = {};
    gcl.handle = gc.handle;
    ioctl(fd, DRM_IOCTL_GEM_CLOSE, &gcl);
    close(fd);
    return 0;
}
