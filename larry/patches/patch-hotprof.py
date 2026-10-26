#!/usr/bin/env python3
# patch-hotprof.py [TIER_CPP] - larry, STRATA_XE_HOT_PROF=1 (diagnostic): where hot_put's time goes - waiting for a ring
# slot's DMA, the CPU copy, waiting for the submit lock, the submission - summed over all threads and printed with the
# prefetch's end (the staging sits at ~14 GB/s whatever the ring / chunk / stagers / spinning / prefetching).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOT_PROF" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""sycl::event hot_put(sycl::queue& q, uint8_t* dst, const uint8_t* src, size_t bytes, bool pinned, uint8_t* ring,
                    std::vector<sycl::event>& ev, int& next) {
    const size_t C = hot_chunk();""", """// larry, STRATA_XE_HOT_PROF: hot_put's parts in us, summed over threads (slot wait, copy, lock wait, submit), bytes
std::atomic<int64_t> g_hp_wait{0}, g_hp_copy{0}, g_hp_lock{0}, g_hp_sub{0}, g_hp_bytes{0}, g_hp_pin{0};
bool hot_prof() { static const bool on = env_i("STRATA_XE_HOT_PROF", 0) != 0; return on; }
sycl::event hot_put(sycl::queue& q, uint8_t* dst, const uint8_t* src, size_t bytes, bool pinned, uint8_t* ring,
                    std::vector<sycl::event>& ev, int& next) {
    const size_t C = hot_chunk();
    const bool hp = hot_prof();
    auto us = [] { return (int64_t) (now_ms() * 1000.0); };""")
rep("""        if (pinned) {
            std::lock_guard<PrioMutex> lk(g_submit);
            last = q.memcpy(dst + off, src + off, nb);
            continue;
        }""", """        if (pinned) {
            const int64_t t0 = hp ? us() : 0;
            std::lock_guard<PrioMutex> lk(g_submit);
            const int64_t t1 = hp ? us() : 0;
            last = q.memcpy(dst + off, src + off, nb);
            if (hp) { g_hp_lock += t1 - t0; g_hp_sub += us() - t1; g_hp_pin += (int64_t) nb; }
            continue;
        }""")
rep("""        ev[(size_t) k].wait();   // a default-constructed event is complete
        uint8_t* slot = ring + (size_t) k * C;
        hot_copy(slot, src + off, nb);
        std::lock_guard<PrioMutex> lk(g_submit);
        last = ev[(size_t) k] = q.memcpy(dst + off, slot, nb);""", """        const int64_t t0 = hp ? us() : 0;
        ev[(size_t) k].wait();   // a default-constructed event is complete
        const int64_t t1 = hp ? us() : 0;
        uint8_t* slot = ring + (size_t) k * C;
        hot_copy(slot, src + off, nb);
        const int64_t t2 = hp ? us() : 0;
        std::lock_guard<PrioMutex> lk(g_submit);
        const int64_t t3 = hp ? us() : 0;
        last = ev[(size_t) k] = q.memcpy(dst + off, slot, nb);
        if (hp) { g_hp_wait += t1 - t0; g_hp_copy += t2 - t1; g_hp_lock += t3 - t2; g_hp_sub += us() - t3; g_hp_bytes += (int64_t) nb; }""")
# print at prefetch_end
rep("""void Ds41XeTier::prefetch_end() {
    Ds41XeImpl& im = *im_;
    if (!im.pf_on) return;""", """void Ds41XeTier::prefetch_end() {
    Ds41XeImpl& im = *im_;
    if (!im.pf_on) return;
    if (hot_prof()) {   // larry, STRATA_XE_HOT_PROF
        std::fprintf(stderr, "xe hot prof: %.2f GB through the rings (%.2f GB pinned direct): slot waits %.0f ms, copies %.0f ms (%.1f GB/s a "
                     "thread), lock waits %.0f ms, submits %.0f ms (thread-ms)\\n", g_hp_bytes.load() / 1e9, g_hp_pin.load() / 1e9,
                     g_hp_wait.load() / 1e3, g_hp_copy.load() / 1e3, g_hp_copy.load() > 0 ? g_hp_bytes.load() / 1e3 / g_hp_copy.load() : 0.0,
                     g_hp_lock.load() / 1e3, g_hp_sub.load() / 1e3);
        g_hp_wait = 0; g_hp_copy = 0; g_hp_lock = 0; g_hp_sub = 0; g_hp_bytes = 0; g_hp_pin = 0;
    }""")
open(P, "w").write(s)
print("patched", P)
