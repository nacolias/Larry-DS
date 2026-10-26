// hs_probe.cpp - larry: can a card wait for the HOST without a host wait or a queue submission at release time?
// (what a host running layers ahead of the dense card needs: the routed sum of a layer waits for the CPU pool's part,
// which the host computes later).  One card (STRATA_HS_DEV, default 1), bounded waits only, no card-to-card copies.
//   A) a Level Zero event from a host-visible pool, wrapped as a SYCL event (make_event, ownership keep): a kernel on
//      an in-order queue depends on it; the host signals it later with zeEventHostSignal.  Checks: the kernel does not
//      run early, the submissions behind it do not block the host, the host-signal -> kernel latency, reuse after
//      zeEventHostReset (a ring of 4), and the signal from another thread while the main thread submits.
//   B) spin kernels polling a pinned host flag the host sets after 2 ms: (1) plain system-scope atomic loads (known:
//      they never see it), (2) loads through an annotated_ptr with L1/L3 uncached read hints, (3) atomic fetch_add(0)
//      (a PCIe atomic).  Each spin is bounded (2^22 iterations, ~1 s) and reports whether it saw the write and when.
// Build: icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" hs_probe.cpp -lze_loader
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
namespace si = sycl::ext::intel::experimental;

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define ZE(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { std::printf("%s = 0x%x\n", #x, (unsigned) r_); std::exit(1); } } while (0)

int main() {
    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
            for (const auto& d : p.get_devices()) if (d.is_gpu()) gpus.push_back(d);
    const int di = std::getenv("STRATA_HS_DEV") ? std::atoi(std::getenv("STRATA_HS_DEV")) : 1;
    if (gpus.empty() || di >= (int) gpus.size()) { std::printf("no level-zero gpu %d\n", di); return 1; }
    sycl::device dev = gpus[(size_t) di];
    sycl::context ctx = dev.get_platform().ext_oneapi_get_default_context();   // the engine's context
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());
    std::printf("device %d: %s\n", di, dev.get_info<sycl::info::device::name>().c_str());
    auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    volatile int* hflag = sycl::malloc_host<int>(64, q);   // [0] kernel ran (A) / host write (B), [16] spin result
    volatile double* hts = (volatile double*) sycl::malloc_host<double>(8, q);
    int* dscratch = sycl::malloc_device<int>(1024, q);
    auto spin_small = [&](int n) {   // n tiny kernels (what the host would queue behind the waiting one)
        for (int i = 0; i < n; ++i) q.single_task([=] { dscratch[i & 1023] += 1; });
    };

    // ---------------- A) host-signalled Level Zero events
    ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 4};
    ze_event_pool_handle_t pool = nullptr;
    ZE(zeEventPoolCreate(zctx, &pd, 1, &zdev, &pool));
    ze_event_handle_t zev[4];
    std::vector<sycl::event> sev;
    for (uint32_t i = 0; i < 4; ++i) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
        ZE(zeEventCreate(pool, &ed, &zev[i]));
        sev.push_back(sycl::make_event<sycl::backend::ext_oneapi_level_zero>({zev[i], sycl::ext::oneapi::level_zero::ownership::keep}, ctx));
    }
    // watchdog: if a step makes no progress for 5 s (the runtime blocking the host on an interop event, a GPU wait that
    // never ends), it signals every event so the queue drains and the probe ends normally (no kill with work queued)
    std::atomic<double> beat{now_us()};
    std::atomic<bool> wd_stop{false};
    std::thread wd([&] {
        while (!wd_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (now_us() - beat.load() > 5e6) {
                std::printf("WATCHDOG: no progress for 5 s - signalling every event\n");
                for (int i = 0; i < 4; ++i) zeEventHostSignal(zev[i]);
                beat = now_us();
            }
        }
    });
    {   // A1: the dependent kernel must not run before the signal; the host must not block submitting behind it
        hflag[0] = 0;
        const double t0 = now_us();
        q.submit([&](sycl::handler& h) {
            h.depends_on(sev[0]);
            h.single_task([=] { hflag[0] = 1; });
        });
        const double t1 = now_us();
        spin_small(100);
        const double t2 = now_us();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int early = hflag[0];
        const double ts = now_us();
        ZE(zeEventHostSignal(zev[0]));
        double seen = 0;
        while (now_us() - ts < 2e6) { if (hflag[0]) { seen = now_us(); break; } }
        q.wait();
        std::printf("A1: submit of the waiting kernel %.1f us, 100 kernels behind it %.1f us (%.1f each); ran before the signal: %s;"
                    " signal -> its write seen %s%.1f us\n", t1 - t0, t2 - t1, (t2 - t1) / 100, early ? "YES (bad)" : "no",
                    seen ? "" : "NEVER ", seen ? seen - ts : 0.0);
    }
    beat = now_us();
    {   // A2: reuse - a ring of 4 events, reset by the host once their waiter is done; 2000 rounds
        std::vector<double> lat;
        for (int r = 0; r < 2000; ++r) {
            const int k = r & 3;
            if (r >= 4) ZE(zeEventHostReset(zev[k]));   // its previous waiter completed (q.wait below)
            hflag[0] = 0;
            q.submit([&](sycl::handler& h) {
                h.depends_on(sev[(size_t) k]);
                h.single_task([=] { hflag[0] = 1; });
            });
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            if (hflag[0]) { std::printf("A2: round %d ran early\n", r); break; }
            const double ts = now_us();
            ZE(zeEventHostSignal(zev[k]));
            while (!hflag[0] && now_us() - ts < 1e6) {}
            lat.push_back(now_us() - ts);
            q.wait();
            beat = now_us();
        }
        std::sort(lat.begin(), lat.end());
        if (!lat.empty())
            std::printf("A2: %zu rounds, signal -> kernel write seen: median %.1f us, p90 %.1f, max %.1f\n", lat.size(), lat[lat.size() / 2],
                        lat[lat.size() * 9 / 10], lat.back());
    }
    beat = now_us();
    {   // A3: the signal from a helper thread while the main thread keeps submitting (the host-ahead shape)
        for (int k = 0; k < 4; ++k) ZE(zeEventHostReset(zev[k]));
        hflag[0] = 0;
        std::atomic<bool> go{false};
        std::thread helper([&] {
            while (!go.load()) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            hts[0] = now_us();
            ZE(zeEventHostSignal(zev[1]));
        });
        q.submit([&](sycl::handler& h) {
            h.depends_on(sev[1]);
            h.single_task([=] { hflag[0] = 1; });
        });
        go = true;
        const double t1 = now_us();
        spin_small(2000);   // ~8 ms of submissions on the main thread
        const double t2 = now_us();
        double seen = 0;
        while (now_us() - t1 < 2e6) { if (hflag[0]) { seen = now_us(); break; } }
        helper.join();
        q.wait();
        std::printf("A3: main thread queued 2000 kernels in %.0f us meanwhile; helper's signal -> write seen %s%.1f us\n", t2 - t1,
                    seen ? "" : "NEVER ", seen ? seen - hts[0] : 0.0);
    }

    beat = now_us();
    // ---------------- B) spin kernels on a pinned host flag
    auto spin_test = [&](const char* what, int mode) {
        hflag[0] = 0;
        hflag[16] = -1;
        hts[1] = 0;
        int* hf = (int*) hflag;
        q.single_task([=] {   // bounded: 2^22 iterations (~1 s)
            int k = 0;
            if (mode == 0) {
                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> f(hf[0]);
                for (; k < (1 << 22) && f.load(sycl::memory_order::acquire) == 0; ++k) {}
            } else if (mode == 1) {
                sx::annotated_ptr<int, decltype(sx::properties(si::read_hint<si::cache_control<si::cache_mode::uncached, sx::cache_level::L1, sx::cache_level::L3>>))> p(hf);
                for (; k < (1 << 22) && p[0] == 0; ++k) {}
            } else {
                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> f(hf[0]);
                for (; k < (1 << 22) && f.fetch_add(0) == 0; ++k) {}
            }
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(hf[16]).store(k);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const double ts = now_us();
        hflag[0] = 1;
        double seen = 0;
        while (now_us() - ts < 3e6) { if (hflag[16] >= 0) { seen = now_us(); break; } }
        q.wait();
        std::printf("B%d %s: %s after %d iterations, host write -> spin end %.1f us\n", mode + 1, what,
                    hflag[16] >= 0 && hflag[16] < (1 << 22) ? "SAW the write" : "did not see it", (int) hflag[16], seen ? seen - ts : -1.0);
    };
    spin_test("atomic loads (system scope)", 0);
    beat = now_us();
    spin_test("annotated_ptr loads, L1/L3 uncached", 1);
    beat = now_us();
    spin_test("atomic fetch_add(0) (PCIe atomic)", 2);
    wd_stop = true;
    wd.join();

    for (int i = 0; i < 4; ++i) zeEventDestroy(zev[i]);
    zeEventPoolDestroy(pool);
    sycl::free((void*) hflag, q);
    sycl::free((void*) hts, q);
    sycl::free(dscratch, q);
    std::printf("done\n");
    return 0;
}
