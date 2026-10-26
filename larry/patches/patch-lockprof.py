#!/usr/bin/env python3
# patch-lockprof.py [TIER_CPP] - larry (diagnostic): STRATA_XE_LOCK_PROF=1 - the tier's submit lock (g_submit, every
# enqueue of the card threads, the stagers, the CPU-light part) per call site: how often it is taken, how long its
# takers waited for it and how long each site held it (total / max), printed every 400 prompt calls.  A short prompt's
# calls are latency-bound (abwT95's perf: the CPU pool's workers spin ~60 % of the time, the card threads are off the
# CPU through their 20-38 ms "GEMM" enqueue phase): who holds the lock while the cards wait?  Sites are return
# addresses: their object and offset (dladdr; addr2line -f names them).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_LOCK_PROF" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""struct PrioMutex {
    std::mutex m;
    std::atomic<int> hi{0};
    void lock() {
        while (hi.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        m.lock();
    }
    void unlock() { m.unlock(); }
};
PrioMutex g_submit;
""", """// larry, STRATA_XE_LOCK_PROF (patch-lockprof.py): per call site (return address) takes, wait and hold times
bool lock_prof_on() { static const bool on = env_i("STRATA_XE_LOCK_PROF", 0) != 0; return on; }
struct LockSite { const void* site; int64_t n; double wait, hold, hold_max; };
std::vector<LockSite> g_lk_sites;            // touched only with g_submit held
thread_local double g_lk_t0 = 0;
thread_local const void* g_lk_site = nullptr;
LockSite& lock_site(const void* site) {
    for (auto& e : g_lk_sites) if (e.site == site) return e;
    g_lk_sites.push_back({site, 0, 0, 0, 0});
    return g_lk_sites.back();
}
struct PrioMutex {
    std::mutex m;
    std::atomic<int> hi{0};
    __attribute__((noinline)) void lock() {
        const bool lp = lock_prof_on();
        const double tw = lp ? now_ms() : 0;
        while (hi.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        m.lock();
        if (lp) {
            g_lk_t0 = now_ms();
            g_lk_site = __builtin_return_address(0);
            LockSite& e = lock_site(g_lk_site);
            ++e.n;
            e.wait += g_lk_t0 - tw;
        }
    }
    void unlock() {
        if (lock_prof_on() && g_lk_site) {
            const double h = now_ms() - g_lk_t0;
            LockSite& e = lock_site(g_lk_site);
            e.hold += h;
            e.hold_max = std::max(e.hold_max, h);
            g_lk_site = nullptr;
        }
        m.unlock();
    }
};
PrioMutex g_submit;
/// the top sites by hold time (call with g_submit held), then zeroed
void lock_prof_dump(const char* what) {
    std::vector<LockSite> v = g_lk_sites;
    std::sort(v.begin(), v.end(), [](const LockSite& a, const LockSite& b) { return a.hold + a.wait > b.hold + b.wait; });
    std::fprintf(stderr, "xe lock prof (%s): site: takes, waited ms, held ms (max)\\n", what);
    for (size_t i = 0; i < v.size() && i < 12; ++i) {
        Dl_info di;
        const char* nm = "?";
        uintptr_t off = 0;
        if (dladdr(v[i].site, &di)) { off = (uintptr_t) v[i].site - (uintptr_t) di.dli_fbase; nm = di.dli_fname ? di.dli_fname : "?"; }   // the offset in its object (addr2line -f)
        std::fprintf(stderr, "  %-60.60s +0x%-6lx %7lld  wait %8.1f  hold %8.1f (%6.2f)\\n", nm, (unsigned long) off, (long long) v[i].n,
                     v[i].wait, v[i].hold, v[i].hold_max);
    }
    g_lk_sites.clear();
}
""")
rep("""#include <immintrin.h>
#endif
""", """#include <immintrin.h>
#endif
#include <dlfcn.h>
""")
# print every 400 prompt calls, at run_big's end (the prof line's place)
# run_big's own end: the last of the four "im.st.wall_ms += ..." returns in the file
tail = """    im.st.wall_ms += now_ms() - t0;
    return true;
}"""
k = s.rfind(tail)
if k < 0 or s.find("bool run_big(") < 0 or k < s.find("bool run_big("):
    sys.exit("run_big end not found")
s = s[:k] + """    im.st.wall_ms += now_ms() - t0;
    if (lock_prof_on()) {   // larry, STRATA_XE_LOCK_PROF
        static int64_t lpn = 0;
        if (++lpn % 400 == 0) { std::lock_guard<PrioMutex> lk(g_submit); lock_prof_dump("400 prompt calls"); }
    }
    return true;
}""" + s[k + len(tail):]
open(P, "w").write(s)
print("patched", P)
