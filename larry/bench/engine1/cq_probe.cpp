// cq_probe.cpp - larry, engine1: does a second in-order queue's host -> device copy (pinned source) run while the
// card's first queue runs a spin kernel?  (The tier's notes: "a second queue of a spinning card does not execute";
// engine1 wants its promotion copies beside the TP passes, which spin at their cross-card waits.)
// Card 1.  Queue A: a bounded spin (~30 ms, a host flag ends it early).  Queue B: a 16 MB memcpy H2D, or a copy KERNEL
// reading pinned memory.  The host times B's completion against A's.  No card-to-card traffic; every wait bounded.
// Build: icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" cq_probe.cpp
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    std::vector<sycl::device> gpus;
    sycl::platform plat;
    for (const auto & p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            auto d = p.get_devices(sycl::info::device_type::gpu);
            if (!d.empty()) { gpus = d; plat = p; break; }
        }
    if (gpus.size() < 2) { std::printf("need 2 gpus\n"); return 1; }
    sycl::context ctx = plat.khr_get_default_context();
    sycl::device dev = gpus[1];
    sycl::queue qa(ctx, dev, sycl::property::queue::in_order()), qb(ctx, dev, sycl::property::queue::in_order());
    const size_t bytes = (size_t) 16 << 20;
    uint8_t * hsrc = sycl::malloc_host<uint8_t>(bytes, ctx);
    uint8_t * ddst = sycl::malloc_device<uint8_t>(bytes, qb);
    int * stop = sycl::malloc_host<int>(16, ctx);
    for (size_t i = 0; i < bytes; ++i) hsrc[i] = (uint8_t) i;
    qb.memcpy(ddst, hsrc, bytes).wait();   // warm both
    qa.single_task([] {}).wait();
    for (int mode = 0; mode < 2; ++mode) {
        for (int rep = 0; rep < 3; ++rep) {
            stop[0] = 0;
            int * sp = stop;
            // A: spin until the host flag (fence + volatile poll: hs_probe2's working form), bounded ~2^16 polls
            const double t0 = now_us();
            sycl::event ea = qa.single_task([=] {
                for (int k = 0; k < (1 << 16); ++k) {
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    if (*(volatile int *) sp) break;
                }
            });
            std::this_thread::sleep_for(std::chrono::microseconds(500));   // A is spinning now
            const double t1 = now_us();
            sycl::event eb;
            if (mode == 0) {
                eb = qb.memcpy(ddst, hsrc, bytes);
            } else {
                using u4 = sycl::vec<uint32_t, 4>;
                const u4 * s = (const u4 *) hsrc;
                u4 * d = (u4 *) ddst;
                eb = qb.parallel_for(sycl::range<1>(bytes / 16), [=](sycl::id<1> i) { d[i] = s[i]; });
            }
            double tb = 0;
            while (now_us() - t1 < 20000) {
                if (eb.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete) { tb = now_us(); break; }
            }
            stop[0] = 1;   // release A
            ea.wait();
            const double ta = now_us();
            eb.wait();
            const double tb2 = now_us();
            std::printf("%s rep %d: B done %s%.0f us after its submit (A spinning); A released at %.0f, B's wait returned %.0f us after submit\n",
                        mode == 0 ? "memcpy H2D (copy engine)" : "copy kernel (pinned reads)", rep, tb ? "" : "NOT within 20 ms - ",
                        tb ? tb - t1 : 20000.0, ta - t0, tb2 - t1);
            std::fflush(stdout);
        }
    }
    sycl::free(hsrc, ctx);
    sycl::free(ddst, qb);
    sycl::free(stop, ctx);
    std::printf("done\n");
    return 0;
}
