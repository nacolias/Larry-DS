#!/usr/bin/env python3
# patch-cpulight2.py [TIER_CPP] - STRATA_XE_CPU_LIGHT's sum without VRAM: the dense card's combine reads the pool's
# weighted sum (pinned host memory) directly instead of a copy in a dense-card buffer.  That buffer (n x 20 KB, ~70 MB
# after a 6.7K prompt) put DSpark's long prompts under ggml-sycl's fp16 weight-cache floor (3 GB free on card 0): their
# dense phase 16.6 -> 18.8 s (31K), 50.1 -> 60.1 s (97K); plain has the headroom.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "straight from pinned host memory" in s:
    print("already patched"); sys.exit(0)
old = """        const float* cpu_part = nullptr;   // STRATA_XE_CPU_LIGHT: the pool's sum, copied onto the dense card first
        if (!light.empty()) {
            std::lock_guard<PrioMutex> lk(g_submit);
            if (!grow_dev(im.lt_dev, im.lt_dev_cap, NE, dc->q)) {
                std::fprintf(stderr, "xe tier: dense card: CPU-light buffer allocation failed\\n");
                return false;
            }
            dc->q.memcpy(im.lt_dev, im.lt_hs, NE * 4);
            cpu_part = im.lt_dev;
        }
"""
new = """        // STRATA_XE_CPU_LIGHT: the pool's sum, read by the combine straight from pinned host memory - a copy in a
        // dense-card buffer (~70 MB after a 6.7K prompt) put DSpark's long prompts under ggml-sycl's fp16 weight-cache
        // floor (3 GB free on card 0): their dense phase +13-20 %
        const float* cpu_part = light.empty() ? nullptr : im.lt_hs;
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
