// pin_probe.cpp - how much SYCL host USM (pinned) can one process allocate on larry, and does a kernel still launch?
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
    const size_t chunk = (size_t) (argc > 1 ? atof(argv[1]) : 2.0) * (1ull << 30);
    const int maxn = argc > 2 ? atoi(argv[2]) : 64;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::vector<void*> v;
    float* d = sycl::malloc_device<float>(1024, q);
    for (int i = 0; i < maxn; ++i) {
        void* p = sycl::malloc_host(chunk, q);
        if (!p) { std::printf("malloc_host failed after %zu GiB\n", v.size() * chunk >> 30); break; }
        std::memset(p, 1, chunk);   // touch
        v.push_back(p);
        try {
            q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> j) { d[j] = (float) j[0]; }).wait();
        } catch (sycl::exception& e) {
            std::printf("kernel failed with %zu GiB pinned: %s\n", v.size() * chunk >> 30, e.what());
            break;
        }
        if ((i + 1) % 4 == 0) std::printf("%zu GiB pinned, kernel ok\n", v.size() * chunk >> 30);
    }
    for (void* p : v) sycl::free(p, q);
    return 0;
}
