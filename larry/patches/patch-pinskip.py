#!/usr/bin/env python3
# patch-pinskip.py [TIER_CPP] - larry, STRATA_XE_PIN_SKIP_GIB (default 0): the first GiB of the ranked arena stay pageable
# and the pinned budget starts after them.  The pinned arena (STRATA_XE_PINNED_GIB, 24) took the hottest arena experts -
# the ~19.7 GiB card 0's expert cache holds anyway, so a prompt streamed almost only pageable experts (through the hot
# ring: the CCD -> IO-die link, ~21.7 GB/s) while pinned ones DMA at the uplink's 28.4 GB/s with no CPU copy.  Skipping
# about the cache's size pins the experts prompts actually stream (and decode's CPU-pool / promotion candidates);
# card 0's seeds and regrows then copy their experts through the bounce instead.  Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "STRATA_XE_PIN_SKIP_GIB" in s:
    print("already patched"); sys.exit(0)
old = """    int64_t pinned = 0;
    std::vector<uint8_t> chunk_pinned(nchunks, 0);
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        if (pinned + (int64_t) chunk_size[c] <= pin_cap) {
"""
new = """    int64_t pinned = 0;
    std::vector<uint8_t> chunk_pinned(nchunks, 0);
    // larry, STRATA_XE_PIN_SKIP_GIB (default 0): the first GiB of the ranked arena - what card 0's expert cache holds
    // anyway - stay pageable; the pinned budget goes to the experts prompts stream and decode promotes
    const int64_t pin_skip = (int64_t) (env_d("STRATA_XE_PIN_SKIP_GIB", 0.0) * kGiB);
    int64_t ranked_bytes = 0;
    for (size_t c = 0; c < nchunks; ++c) {
        uint8_t* p = nullptr;
        const bool skip = ranked_bytes < pin_skip;
        ranked_bytes += (int64_t) chunk_size[c];
        if (!skip && pinned + (int64_t) chunk_size[c] <= pin_cap) {
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
