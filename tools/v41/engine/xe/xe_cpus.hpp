// tools/v41/engine/xe/xe_cpus.hpp - larry: host thread placement for the TP passes and the tier's helpers.
//
// larry's Ryzen 7 5800X: 8 cores x 2 threads, CPU n and CPU n + 8 one core's two threads.  The CPU expert pool pins
// its 7 workers to CPUs 1-7 (cores 1-7); everything else floated, and with the workers spinning on their CPUs the
// floating threads - the TP pass's host loop (the mailbox spin, the pool's host share of the work), the promotion
// fill helpers, the Engram reader - landed on the workers' SMT siblings (Strata measured its pool at 36.3 GB/s alone,
// 26.9 inside a host loop that floated).  STRATA_HOST_CPUS / STRATA_XE_HELPER_CPUS place them ("0", "8", "7,15",
// "8-15"); unset: as before.
#pragma once

#include <pthread.h>
#include <sched.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::v41::xe {

/// "0,8" / "8-15" / "7,15" -> a CPU set; false when the list is empty or malformed
inline bool parse_cpus(const char* s, cpu_set_t& set) {
    CPU_ZERO(&set);
    if (!s || !*s) return false;
    int n = 0;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        const long a = std::strtol(p, &e, 10);
        if (e == p || a < 0 || a >= CPU_SETSIZE) return false;
        long b = a;
        p = e;
        if (*p == '-') {
            b = std::strtol(p + 1, &e, 10);
            if (e == p + 1 || b < a || b >= CPU_SETSIZE) return false;
            p = e;
        }
        for (long c = a; c <= b; ++c) { CPU_SET((int) c, &set); ++n; }
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return n > 0;
}

/// The CPU mask of the first caller (call it before any thread pins itself: the tier's / TP core's init)
inline const cpu_set_t& process_cpus() {
    static const cpu_set_t s = [] {
        cpu_set_t x;
        CPU_ZERO(&x);
        if (sched_getaffinity(0, sizeof x, &x) != 0) for (int c = 0; c < 64; ++c) CPU_SET(c, &x);
        return x;
    }();
    return s;
}

/// A helper thread's placement: STRATA_XE_HELPER_CPUS, else the process's original mask (a thread started by a pinned
/// thread would otherwise inherit its CPU)
inline void place_helper(pthread_t t) {
    static const char* spec = std::getenv("STRATA_XE_HELPER_CPUS");
    cpu_set_t s;
    if (!parse_cpus(spec, s)) s = process_cpus();
    pthread_setaffinity_np(t, sizeof s, &s);
}

/// Pins the calling thread to STRATA_HOST_CPUS for its lifetime (restored at destruction); a no-op when unset
struct HostPin {
    cpu_set_t prev;
    bool on = false;
    HostPin() {
        static const char* spec = std::getenv("STRATA_HOST_CPUS");
        cpu_set_t s;
        if (!parse_cpus(spec, s)) return;
        if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return;
        on = pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
    }
    ~HostPin() {
        if (on) pthread_setaffinity_np(pthread_self(), sizeof prev, &prev);
    }
    HostPin(const HostPin&) = delete;
    HostPin& operator=(const HostPin&) = delete;
};

}  // namespace strata::v41::xe
