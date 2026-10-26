#!/usr/bin/env python3
# patch-hotpinwait.py [TIER_CPP] - larry, STRATA_XE_HOT_PINWAIT=1 (default 0 until measured): a pinned expert's chunks
# take the ring's slots too (each waits for that slot's previous DMA; no copy).  hot_put put a pinned expert's ~10 MB
# on the card's copy engine at once, and a card's copies run FIFO across its queues - the other stagers' ring chunks
# queued behind it (a chunk's DMA ~0.6 ms after its submit, STRATA_XE_HOT_PROF) while their lines aged out of the L3:
# DRAM traffic in a prompt is ~2.3 bytes a staged byte (bench/speed/dram-mon.sh), where a ring served from the L3 is
# one.  With the slots a stager has at most STRATA_XE_HOTRING_N chunks in flight, pinned or not.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOT_PINWAIT" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""        if (pinned) {
            const int64_t t0 = hp ? us() : 0;
            std::lock_guard<PrioMutex> lk(g_submit);
            const int64_t t1 = hp ? us() : 0;
            last = q.memcpy(dst + off, src + off, nb);
            if (hp) { g_hp_lock += t1 - t0; g_hp_sub += us() - t1; g_hp_pin += (int64_t) nb; }
            continue;
        }""", """        if (pinned) {
            // larry, STRATA_XE_HOT_PINWAIT (patch-hotpinwait.py): the chunk takes a slot (its previous DMA done first)
            static const bool pin_wait = env_i("STRATA_XE_HOT_PINWAIT", 0) != 0;
            int pk = -1;
            if (pin_wait) {
                pk = next;
                next = (pk + 1) % R;
                const int64_t tw = hp ? us() : 0;
                ev[(size_t) pk].wait();
                if (hp) g_hp_wait += us() - tw;
            }
            const int64_t t0 = hp ? us() : 0;
            std::lock_guard<PrioMutex> lk(g_submit);
            const int64_t t1 = hp ? us() : 0;
            last = q.memcpy(dst + off, src + off, nb);
            if (pk >= 0) ev[(size_t) pk] = last;
            if (hp) { g_hp_lock += t1 - t0; g_hp_sub += us() - t1; g_hp_pin += (int64_t) nb; }
            continue;
        }""")
open(P, "w").write(s)
print("patched", P)
