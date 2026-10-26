// reg_acct.cpp - larry: does memory registered with prepare_for_device_copy (all four cards' context) count as
// "hidden" (GPU-driver / TTM) memory?  /proc/meminfo before and after registering G GiB of THP pageable memory.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void mem(const char* tag) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    long v;
    std::string u;
    long tot = 0, avail = 0, anon = 0, unevict = 0, mlocked = 0;
    while (f >> k >> v) {
        std::getline(f, u);
        if (k == "MemTotal:") tot = v;
        if (k == "MemAvailable:") avail = v;
        if (k == "AnonPages:") anon = v;
        if (k == "Unevictable:") unevict = v;
        if (k == "Mlocked:") mlocked = v;
    }
    std::printf("%-26s avail %6.1f GiB  anon %6.1f  hidden %6.1f  unevictable %5.1f  mlocked %5.1f\n", tag, avail / 1048576.0,
                anon / 1048576.0, (tot - avail - anon) / 1048576.0, unevict / 1048576.0, mlocked / 1048576.0);
}

int main(int argc, char** argv) {
    const size_t G = (size_t) (argc > 1 ? std::atoi(argv[1]) : 4) << 30;
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::context ctx(gpus);
    sycl::queue q1(ctx, gpus[1], sycl::property::queue::in_order());
    mem("start");
    void* p = nullptr;
    if (posix_memalign(&p, 2u << 20, G) != 0) return 1;
    madvise(p, G, MADV_HUGEPAGE);
    std::memset(p, 5, G);
    mem("after touching");
    const double t0 = now_ms();
    const size_t ch = 256u << 20;
    for (size_t off = 0; off < G; off += ch) sx::prepare_for_device_copy((uint8_t*) p + off, ch, ctx);
    const double t1 = now_ms();
    mem("after registering");
    uint8_t* d = sycl::malloc_device<uint8_t>(ch, q1);
    for (size_t off = 0; off < G; off += ch) q1.memcpy(d, (uint8_t*) p + off, ch);
    q1.wait();
    const double t2 = now_ms();
    mem("after DMA to card 1");
    for (size_t off = 0; off < G; off += ch) sx::release_from_device_copy((uint8_t*) p + off, ctx);
    const double t3 = now_ms();
    mem("after release");
    std::printf("register %.0f ms, DMA %.1f GB/s, release %.0f ms (%zu GiB)\n", t1 - t0, G / (t2 - t1) / 1e6, t3 - t2, G >> 30);
    sycl::free(d, q1);
    free(p);
    return 0;
}
