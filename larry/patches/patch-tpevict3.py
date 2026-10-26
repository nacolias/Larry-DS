#!/usr/bin/env python3
# patch-tpevict3.py [TIER_CPP] - larry, after patch-tpevict2.py: evict_slabs waits for the driver to report each freed
# slab before it decides on the next one.  The xe driver shows freed VRAM a little later, so the loop's free-VRAM test
# still saw the old value and every card gave all three of its evictable slabs back at 594K tokens of context (one or
# two were enough).  Now: after a slab is freed, poll until the free VRAM rises by ~the slab (or stops changing, <= 1 s),
# then test again.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "until the driver shows this slab" in s:
    print("already patched"); sys.exit(0)
old = """        set.resize(k);
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
    }
    if (!n) return 0;
    // the driver may report the freed pages a little later: until it does or the reading stops rising
    for (int i = 0, same = 0; i < 100 && c.free_vram() < need && same < 3; ++i) {
        const int64_t f0 = c.free_vram();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        same = c.free_vram() == f0 ? same + 1 : 0;
    }
"""
new = """        set.resize(k);
        const int64_t fr0 = c.free_vram();
        local_free(c.q, last);
        c.slabs.pop_back();
        c.slab_used = kSlab;   // the slabs left are full: the next expert opens a new one
        // the xe driver shows freed VRAM a little later: wait until the driver shows this slab (or the reading stops
        // rising) before the next test - else every evictable slab went at once
        for (int i = 0, same = 0; i < 50 && c.free_vram() < fr0 + (int64_t) kSlab * 9 / 10 && same < 5; ++i) {
            const int64_t f0 = c.free_vram();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            same = c.free_vram() == f0 ? same + 1 : 0;
        }
    }
    if (!n) return 0;
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
