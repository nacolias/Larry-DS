#!/usr/bin/env python3
# patch-hotspin.py [TIER_CPP] - larry, STRATA_XE_HOTRING_SPIN=N (default 0 = off): a hot-ring stager waiting for its
# slot's previous DMA polls the event (N spins of _mm_pause between status queries, then the blocking wait) instead of
# blocking at once.  The prefetch's staging runs ~14 GB/s with 4 stagers whatever the ring's depth or chunk size
# (abwT33) - ~3.5 GB/s a thread, the copy itself does ~10; bench/dqb/stage_ring reached 21.7 with a submit thread
# that polls.  The blocking wait may sleep and wake late every 1 MiB chunk.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_HOTRING_SPIN" in s:
    print("already patched"); sys.exit(0)
old = """        ev[(size_t) k].wait();   // a default-constructed event is complete
"""
new = """        // larry, STRATA_XE_HOTRING_SPIN: poll first (a blocking wait may sleep and wake late, every chunk)
        static const int hot_spin = env_i("STRATA_XE_HOTRING_SPIN", 0);
        if (hot_spin > 0)
            for (int it = 0; it < (1 << 20) && ev[(size_t) k].get_info<sycl::info::event::command_execution_status>() !=
                                                   sycl::info::event_command_status::complete; ++it)
                for (int sp = 0; sp < hot_spin; ++sp) _mm_pause();
        ev[(size_t) k].wait();   // a default-constructed event is complete
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
