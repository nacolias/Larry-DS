#!/usr/bin/env python3
# patch-pfbtrace.py [TIER_CPP] - larry (diagnostic, after patch-barstage2.py): with BAR staging prefetch_begin took
# 481-699 ms a batch instead of 32-124 (abwT84, STRATA_BATCH_TRACE).  STRATA_XE_ALLOC_LOG=1 now also prints its parts:
# the promotion drop, each card's flex region (kept / re-made and how long), the fault-in, the call buffers.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
if "patch-pfbtrace" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    if (!on || !im.pf_stage_cap || im.pf_on || last < first) return false;
    pc_drop(im);   // the promoted experts' VRAM back first (the buffers below are sized from what is free)
""", """    if (!on || !im.pf_stage_cap || im.pf_on || last < first) return false;
    // larry, patch-pfbtrace (STRATA_XE_ALLOC_LOG=1): prefetch_begin's parts
    static const bool pfb_log = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;
    const double pfb0 = now_ms();
    std::string pfb;
    pc_drop(im);   // the promoted experts' VRAM back first (the buffers below are sized from what is free)
    if (pfb_log) pfb += "pc_drop " + std::to_string((int) (now_ms() - pfb0)) + " ms";
""")
rep("""        if (!c.flex || c.flex_cap < 2 * cap || c.flex_cap > 2 * cap + ((size_t) 64 << 20)) {
            if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (!c.flex) { c.pbuf_cap = 0; continue; }
            if (bar_cpu(c.flex)) touch.push_back(&c);
        }
""", """        if (!c.flex || c.flex_cap < 2 * cap || c.flex_cap > 2 * cap + ((size_t) 64 << 20)) {
            const double ta = now_ms();
            const size_t was = c.flex_cap;
            if (c.flex) { c.q.wait(); local_free(c.q, c.flex); }
            c.flex = card_alloc(c.q, 2 * cap);   // larry, patch-barstage
            c.flex_cap = c.flex ? 2 * cap : 0;
            if (pfb_log) {
                char b[160];
                std::snprintf(b, sizeof b, ", card %d flex %.2f -> %.2f GiB %s %.0f ms", c.idx, was / kGiB, c.flex_cap / kGiB,
                              c.flex && bar_cpu(c.flex) ? "BAR" : "local", now_ms() - ta);
                pfb += b;
            }
            if (!c.flex) { c.pbuf_cap = 0; continue; }
            if (bar_cpu(c.flex)) touch.push_back(&c);
        } else if (pfb_log) {
            pfb += ", card " + std::to_string(c.idx) + " flex kept";
        }
        if (pfb_log && keep_st) pfb += " (stream kept)";
""")
rep("""        for (auto& t : th) t.join();
        static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;""", """        for (auto& t : th) t.join();
        if (pfb_log) pfb += ", fault-in done at " + std::to_string((int) (now_ms() - pfb0)) + " ms";
        static const bool all = env_i("STRATA_XE_ALLOC_LOG", 0) != 0;""")
rep("""    static bool logged = false;
    if (!logged || xe_prof()) {
        logged = true;
        std::fprintf(stderr, "xe prefetch: layers %lld..%lld, %d tokens a call, largest layer %.2f GiB; buffers per card 2 x", (long long) first,""",
    """    if (pfb_log)
        std::fprintf(stderr, "xe prefetch begin: %s; call buffers etc. done at %.0f ms\\n", pfb.c_str(), now_ms() - pfb0);
    static bool logged = false;
    if (!logged || xe_prof()) {
        logged = true;
        std::fprintf(stderr, "xe prefetch: layers %lld..%lld, %d tokens a call, largest layer %.2f GiB; buffers per card 2 x", (long long) first,""")
open(P, "w").write(s)
print("patched", P)
