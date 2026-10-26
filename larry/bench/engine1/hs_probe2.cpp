// hs_probe2.cpp - larry, engine1: which host -> GPU flag does a spinning kernel see, in the TP core's setting (every
// card in the platform's default context, the spin on card 0 or 1, eager or inside a replayed SYCL graph)?
// (In the engine a spin on a sycl::malloc_host flag ran to its bound: tp_core's Engram gate; hs_probe B1 saw one.)
//   memory: M0 sycl::malloc_host, M1 zeMemAllocHost BIAS_UNCACHED, M2 zeMemAllocHost BIAS_CACHED
//   spin:   S0 atomic_ref<system> acquire loads, S1 acquire fence (system) + volatile load each poll
// Each case: the kernel spins (bounded 2^22 polls), the host writes the flag 2 ms after the launch, the kernel stores
// its poll count to a host word (GPU -> host works); prints SEEN / NOT and the latency.  Bounded, no card-to-card copies.
// Build: icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" hs_probe2.cpp -lze_loader
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    std::vector<sycl::device> gpus;
    sycl::platform plat;
    for (const auto & p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            auto d = p.get_devices(sycl::info::device_type::gpu);
            if (!d.empty()) { gpus = d; plat = p; break; }
        }
    if (gpus.empty()) { std::printf("no gpu\n"); return 1; }
    sycl::context ctx = plat.khr_get_default_context();
    std::vector<sycl::queue> qs;
    for (auto & d : gpus) qs.emplace_back(ctx, d, sycl::property::queue::in_order());
    std::printf("%zu GPUs in the default context\n", gpus.size());
    ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto zalloc = [&](ze_host_mem_alloc_flags_t fl) -> void * {
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, fl};
        void * p = nullptr;
        if (zeMemAllocHost(zc, &d, 4096, 4096, &p) != ZE_RESULT_SUCCESS) return nullptr;
        return p;
    };
    void * mem[3] = { sycl::malloc_host(4096, ctx), zalloc(ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED), zalloc(ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED) };
    const char * mname[3] = { "M0 sycl::malloc_host", "M1 ze host UNCACHED", "M2 ze host CACHED" };
    int * done = sycl::malloc_host<int>(16, ctx);   // the kernel's result (GPU -> host)
    for (int m = 0; m < 3; ++m) {
        if (!mem[m]) { std::printf("%s: allocation failed\n", mname[m]); continue; }
        for (int card : {0, 1}) {
            if (card >= (int) qs.size()) continue;
            for (int sp = 0; sp < 2; ++sp) {
                for (int gmode = 0; gmode < 2; ++gmode) {
                    volatile int * flag = (volatile int *) mem[m];
                    flag[0] = 0;
                    done[0] = -1;
                    int * fp = (int *) mem[m];
                    int * dp = done;
                    const int spin = sp;
                    auto emit = [&](sycl::queue & q) {
                        q.single_task([=] {
                            int k = 0;
                            if (spin == 0) {
                                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space> f(fp[0]);
                                for (; k < (1 << 22) && f.load(sycl::memory_order::acquire) == 0; ++k) {}
                            } else {
                                for (; k < (1 << 22); ++k) {
                                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                                    if (*(volatile int *) fp != 0) break;
                                }
                            }
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(dp[0]).store(k);
                        });
                    };
                    sycl::queue & q = qs[(size_t) card];
                    std::unique_ptr<sx::command_graph<sx::graph_state::executable>> ex;
                    if (gmode) {
                        sx::command_graph<sx::graph_state::modifiable> g(ctx, gpus[(size_t) card]);
                        g.begin_recording(q);
                        emit(q);
                        g.end_recording(q);
                        ex = std::make_unique<sx::command_graph<sx::graph_state::executable>>(g.finalize());
                        q.ext_oneapi_graph(*ex);
                    } else {
                        emit(q);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    const double ts = now_us();
                    flag[0] = 1;
                    double seen = 0;
                    while (now_us() - ts < 20e6) { if (*(volatile int *) &done[0] >= 0) { seen = now_us(); break; } }
                    q.wait();
                    const int k = done[0];
                    std::printf("%s card %d %s %s: %s after %d polls, host write -> end %.1f us\n", mname[m], card,
                                spin ? "S1 fence+volatile" : "S0 atomic acquire", gmode ? "graph" : "eager",
                                k >= 0 && k < (1 << 22) ? "SEEN" : "NOT SEEN", k, seen ? seen - ts : -1.0);
                    std::fflush(stdout);
                }
            }
        }
    }
    std::printf("done\n");
    return 0;
}
