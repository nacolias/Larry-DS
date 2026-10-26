// bar_leak.cpp - larry: does freeing a BAR buffer (patch-barstage's bar_alloc / bar_free: an xe buffer object, mmap'd,
// PRIME-exported and imported into Level Zero) give its VRAM back?  abwT80's check config lost ~1 GiB of free VRAM on
// cards 1-2 over 12 prompts (the stream buffers are freed and re-made every prompt).  20 rounds of 256 MiB on card 1 per
// variant, the card's free VRAM printed after each free:
//   0  as patch-barstage (resident, dma-buf fd closed after the import, zeMemFree, munmap, GEM_CLOSE)
//   1  zeContextEvictMemory before zeMemFree
//   2  never made resident
//   3  the dma-buf fd kept open until after zeMemFree
// Run with the service stopped: bar_leak [variant ...]
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <drm/drm.h>
#include <drm/xe_drm.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* PCI = "0000:1b:00.0";
    const size_t S = 256u << 20;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::context ctx(devs);
    int k1 = -1;
    for (size_t i = 0; i < devs.size(); ++i)
        if (devs[i].get_info<sycl::ext::intel::info::device::pci_address>() == PCI) k1 = (int) i;
    if (k1 < 0) return 1;
    sycl::device& dev = devs[(size_t) k1];
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());
    std::string node;
    if (DIR* d = opendir((std::string("/sys/bus/pci/devices/") + PCI + "/drm").c_str())) {
        while (dirent* e = readdir(d)) if (!std::strncmp(e->d_name, "renderD", 7)) node = std::string("/dev/dri/") + e->d_name;
        closedir(d);
    }
    const int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
    drm_xe_device_query dq = {};
    dq.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
    ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq);
    std::vector<uint8_t> qb(dq.size);
    dq.data = (uintptr_t) qb.data();
    ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq);
    const auto* mr = (const drm_xe_query_mem_regions*) qb.data();
    int inst = -1;
    for (uint32_t i = 0; i < mr->num_mem_regions; ++i) if (mr->mem_regions[i].mem_class == DRM_XE_MEM_REGION_CLASS_VRAM) inst = mr->mem_regions[i].instance;
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    auto freeg = [&] { return dev.get_info<sycl::ext::intel::info::device::free_memory>() / 1073741824.0; };
    std::vector<int> vars;
    for (int i = 1; i < argc; ++i) vars.push_back(std::atoi(argv[i]));
    if (vars.empty()) vars = {0, 1, 2, 3};
    for (int v : vars) {
        const double f0 = freeg();
        double fmin = f0;
        for (int r = 0; r < 20; ++r) {
            drm_xe_gem_create gc = {};
            gc.size = S; gc.placement = 1u << inst; gc.flags = DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM; gc.cpu_caching = DRM_XE_GEM_CPU_CACHING_WC;
            if (ioctl(fd, DRM_IOCTL_XE_GEM_CREATE, &gc)) { std::perror("create"); return 1; }
            drm_xe_gem_mmap_offset mo = {}; mo.handle = gc.handle;
            ioctl(fd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mo);
            uint8_t* cpu = (uint8_t*) mmap(nullptr, S, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mo.offset);
            drm_prime_handle ph = {}; ph.handle = gc.handle; ph.flags = DRM_CLOEXEC | DRM_RDWR;
            ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph);
            ze_external_memory_import_fd_t imp = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr, ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, ph.fd};
            ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &imp, 0, 0};
            void* p = nullptr;
            if (zeMemAllocDevice(zc, &dd, S, 4096, zd, &p) != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "import failed\n"); return 1; }
            if (v != 2) zeContextMakeMemoryResident(zc, zd, p, S);
            if (v != 3) close(ph.fd);
            for (size_t o = 0; o < S; o += 65536) cpu[o] = (uint8_t) r;
            q.memset(p, 1, 4096).wait();
            fmin = std::min(fmin, freeg());
            if (v == 1) zeContextEvictMemory(zc, zd, p, S);
            zeMemFree(zc, p);
            if (v == 3) close(ph.fd);
            munmap(cpu, S);
            drm_gem_close cl = {}; cl.handle = gc.handle;
            ioctl(fd, DRM_IOCTL_GEM_CLOSE, &cl);
        }
        q.wait();
        std::printf("variant %d: free VRAM %.3f GiB before, %.3f lowest while held, %.3f after 20 rounds (%+.3f)\n", v, f0, fmin, freeg(),
                    freeg() - f0);
        std::fflush(stdout);
    }
    close(fd);
    return 0;
}
