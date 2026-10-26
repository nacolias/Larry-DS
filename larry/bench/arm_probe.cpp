// arm_probe.cpp - the STRATA_XE_ARM pattern alone: per expert card a spin kernel on a pinned release counter, a
// memcpy and a kernel; the host releases each after a delay.  Counts timeouts and how fast a release is seen.
#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int mode = argc > 2 ? std::atoi(argv[2]) : 0;   // 0: system atomic load; 1: volatile load; 2: device flag, host memcpy on a 2nd queue
    std::vector<sycl::device> devs;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    sycl::context ctx = devs[0].get_platform().khr_get_default_context();
    std::vector<sycl::queue> qs, rq;
    for (auto& d : devs) { qs.emplace_back(ctx, d, sycl::property::queue::in_order()); rq.emplace_back(ctx, d, sycl::property::queue::in_order()); }
    std::vector<int*> dflag(devs.size()), hrel(devs.size());
    const int NC = (int) devs.size();
    std::vector<int*> flag(NC), seen(NC), done(NC);
    std::vector<int> rel(NC, 0);
    std::vector<float*> buf(NC);
    ze_context_handle_t zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    for (int c = 1; c < NC; ++c) {
        if (mode == 3) {   // uncached host memory through Level Zero
            ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
            void* p = nullptr;
            ze_result_t r = zeMemAllocHost(zctx, &hd, 64, 64, &p);
            if (r != ZE_RESULT_SUCCESS) { std::printf("zeMemAllocHost failed %d\n", (int) r); return 1; }
            flag[c] = (int*) p;
        } else flag[c] = sycl::malloc_host<int>(1024, qs[c]);   // a page of its own: nothing the GPU writes shares its line
        *flag[c] = 0;
        done[c] = sycl::malloc_host<int>(1024, qs[c]); *done[c] = 0;
        seen[c] = sycl::malloc_device<int>(1, qs[c]); qs[c].memset(seen[c], 0, 4).wait();
        buf[c] = sycl::malloc_device<float>(1 << 16, qs[c]);
        dflag[c] = sycl::malloc_device<int>(1, qs[c]); qs[c].memset(dflag[c], 0, 4).wait();
        hrel[c] = sycl::malloc_host<int>(64, qs[c]);
    }
    int* err = sycl::malloc_host<int>(1, qs[0]); *err = 0;
    int* fpage = sycl::malloc_host<int>(1024, qs[0]);   // modes 5/6: card c spins on fpage[16 c]
    int* fsrc = sycl::malloc_host<int>(1024, qs[0]);    // the host's release values, copied / stored by card 0
    for (int i = 0; i < 1024; ++i) { fpage[i] = 0; fsrc[i] = 0; }
    sycl::queue relq(ctx, devs[0], sycl::property::queue::in_order());
    int* dbg = sycl::malloc_host<int>(64, qs[0]); for (int i = 0; i < 64; ++i) dbg[i] = -7;
    int timeouts = 0; double seen_sum = 0; int nseen = 0; std::vector<double> lat;
    for (int it = 0; it < iters; ++it) {
        for (int c = 1; c < NC; ++c) {
            int* f = (mode == 2 || mode == 7) ? dflag[c] : mode >= 5 ? fpage + 16 * c : flag[c]; int* s = seen[c]; int* db = dbg + 8 * c; int* e = err; int* dn = done[c]; float* b = buf[c];
            qs[c].single_task([=] {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sn(*s);
                const int want = sn.load() + 1;
                int k = 0;
                if (mode == 2 || mode == 7) {
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> fa(*f);
                    for (; k < (1 << 22) && fa.load(sycl::memory_order::acquire) < want; ++k) {}
                } else if (mode == 0 || mode == 3 || mode == 4 || mode >= 5) {
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> fa(*f);
                    for (; k < (1 << 22) && fa.load(sycl::memory_order::acquire) < want; ++k) {}
                } else {
                    for (; k < (1 << 22) && *(volatile int*) f < want; ++k) {}
                }
                if (k == (1 << 22)) sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(*e).fetch_add(1);
                db[0] = want; db[1] = k; db[2] = *(volatile int*) f;
                sn.store(want);
            });
            qs[c].memset(b, 0, 4096);
            qs[c].single_task([=] { sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(*dn).store(1); });
        }
        // the host: ~300 us of "attention", then the releases
        const double t0 = now_us();
        while (now_us() - t0 < 300) {}
        for (int c = 1; c < NC; ++c) { *(volatile int*) done[c] = 0; }
        const double tr = now_us();
        if (mode == 2) {
            for (int c = 1; c < NC; ++c) { hrel[c][(it & 31) * 2] = ++rel[c]; rq[c].memcpy(dflag[c], &hrel[c][(it & 31) * 2], 4); }
        } else if (mode == 7) {   // card 0's queue copies each card's release into that card's own VRAM (P2P write)
            for (int c = 1; c < NC; ++c) { fsrc[16 * c] = ++rel[c]; relq.memcpy(dflag[c], fsrc + 16 * c, 4); }
        } else if (mode == 5) {   // card 0's copy engine writes all the cards' words at once
            for (int c = 1; c < NC; ++c) fsrc[16 * c] = ++rel[c];
            relq.memcpy(fpage, fsrc, 16 * 4 * NC);
        } else if (mode == 6) {   // a one-lane kernel on card 0 stores them (system scope)
            for (int c = 1; c < NC; ++c) fsrc[16 * c] = ++rel[c];
            int* fp = fpage; int* fs = fsrc; const int nc = NC;
            relq.single_task([=] {
                for (int c = 1; c < nc; ++c)
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(fp[16 * c])
                        .store(*(volatile int*) (fs + 16 * c), sycl::memory_order::release);
            });
        } else {
            for (int c = 1; c < NC; ++c) {
                __atomic_store_n(flag[c], ++rel[c], __ATOMIC_RELEASE);
                if (mode >= 4) { _mm_clflush(flag[c]); _mm_mfence(); }   // the GPU's reads do not snoop: push the line to DRAM
            }
        }
        for (int c = 1; c < NC; ++c) {
            while (__atomic_load_n(done[c], __ATOMIC_ACQUIRE) == 0 && now_us() - tr < 2e6) {}
            if (it >= 20) { seen_sum += now_us() - tr; ++nseen; lat.push_back(now_us() - tr); }
        }
        for (int c = 1; c < NC; ++c) qs[c].wait();
        if (it < 3) for (int c = 1; c < NC; ++c) std::printf("  iter %d card %d: want %d, spun %d, flag seen %d, host flag %d\n", it, c, dbg[8*c], dbg[8*c+1], dbg[8*c+2], (mode == 2 || mode == 7) ? -1 : mode >= 5 ? *(volatile int*) (fpage + 16 * c) : *(volatile int*) flag[c]);
        if (*(volatile int*) err != timeouts) { timeouts = *(volatile int*) err; std::printf("iter %d: timeouts now %d\n", it, timeouts); if (timeouts > 3) break; }
    }
    std::sort(lat.begin(), lat.end());
    if (!lat.empty()) std::printf("latency (after warm-up): p10 %.1f, p50 %.1f, p90 %.1f, max %.1f us\n", lat[lat.size() / 10], lat[lat.size() / 2], lat[lat.size() * 9 / 10], lat.back());
    std::printf("mode %d: %d iterations, timeouts %d, release -> card done (spin exit + memset + flag) mean %.1f us\n", mode, iters, *err, seen_sum / std::max(1, nseen));
    return 0;
}
