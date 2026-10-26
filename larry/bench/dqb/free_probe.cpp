// free_probe.cpp - larry: does the free-VRAM query (ext_intel_free_memory) see a zeMemAllocDevice allocation at once,
// or only once it is touched?  vs sycl::malloc_device
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cstdio>
int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto fr = [&] { return q.get_device().get_info<sycl::ext::intel::info::device::free_memory>() / 1073741824.0; };
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    std::printf("free at start %.2f GiB\n", fr());
    ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void* p = nullptr;
    zeMemAllocDevice(zc, &d, 4ull << 30, 4096, zd, &p);
    std::printf("after native 4 GiB: free %.2f GiB\n", fr());
    const ze_result_t r = zeContextMakeMemoryResident(zc, zd, p, 4ull << 30);
    std::printf("after zeContextMakeMemoryResident (this device): rc %d, free %.2f GiB\n", (int) r, fr());
    void* big = nullptr;
    zeMemAllocDevice(zc, &d, 40ull << 30, 4096, zd, &big);
    const ze_result_t r2 = big ? zeContextMakeMemoryResident(zc, zd, big, 40ull << 30) : ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    std::printf("40 GiB native: ptr %p, resident rc %d (0x%x)\n", big, (int) r2, (unsigned) r2);
    if (big) zeMemFree(zc, big);
    q.memset(p, 0, 4ull << 30).wait();
    std::printf("after touching it: free %.2f GiB\n", fr());
    void* s = sycl::malloc_device(4ull << 30, q);
    std::printf("after sycl 4 GiB: free %.2f GiB\n", fr());
    q.memset(s, 0, 4ull << 30).wait();
    std::printf("after touching it: free %.2f GiB\n", fr());
    zeMemFree(zc, p);
    sycl::free(s, q);
    std::printf("after both freed: free %.2f GiB\n", fr());
    return 0;
}
