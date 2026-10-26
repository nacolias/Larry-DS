#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t SZ = (size_t) std::atol(argv[1]) << 20;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    uint8_t* h = sycl::malloc_host<uint8_t>(SZ, q);
    uint8_t* d = sycl::malloc_device<uint8_t>(SZ, q);
    std::printf("%p %p\n", (void*) h, (void*) d);
    std::memset(h, 1, SZ);
    q.memcpy(d, h, (size_t) 10 << 20).wait();
    std::printf("ok %zu MiB\n", SZ >> 20);
}
