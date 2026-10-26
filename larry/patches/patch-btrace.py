#!/usr/bin/env python3
# patch-btrace.py [GENERATE_CPP] - larry, STRATA_BATCH_TRACE=1 (diagnostic, default off): a served prompt's steps outside
# the layer pipeline, "batch trace: <ms since the request's keep-cache step> <step>" - the keep-cache shrink, the feed,
# per layer-major batch its columns / begin_batch, the prefetch setup, the pipeline, prefetch end, the logits, then the
# regrow (a 16K prompt spent ~1.2 s, a 31K ~1.9 s outside its pipelines).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_BATCH_TRACE" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
""", """double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// larry, STRATA_BATCH_TRACE=1: a served prompt's steps, ms since its keep-cache step (btr_reset)
static double g_btr0 = 0;
static bool btr_on() { static const bool on = std::getenv("STRATA_BATCH_TRACE") != nullptr; return on; }
static void btr_reset() { g_btr0 = now_ms(); }
static void btr(const char * what, int a = -1) {
    if (!btr_on()) return;
    if (a >= 0) std::fprintf(stderr, "batch trace: %9.1f  %s %d\\n", now_ms() - g_btr0, what, a);
    else std::fprintf(stderr, "batch trace: %9.1f  %s\\n", now_ms() - g_btr0, what);
}
""")
# the serve loop
rep("""            bool kept = false;
            if (Ds41Tier::keep_cache_mode() && !by_step && dh >= 2 && !(defer_vram && cache_open))""",
    """            bool kept = false;
            btr_reset();   // larry, STRATA_BATCH_TRACE
            if (Ds41Tier::keep_cache_mode() && !by_step && dh >= 2 && !(defer_vram && cache_open))""")
rep("""            tier.rebalance_begin_prompt();
            tier.reset_stats();
""", """            btr("keep-cache / release", kept ? 1 : 0);
            tier.rebalance_begin_prompt();
            tier.reset_stats();
""")
rep("""            const double tph0 = now_ms();
""", """            const double tph0 = now_ms();
            btr("feed start (batch width)", pb_cap);
""")
rep("""            bcap_n = 0;
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
""", """            bcap_n = 0;
            btr("feed end");
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
""")
rep("""            double ph_feed = now_ms() - tph0, ph_open = 0, ph_tail = 0;   // the request's prompt phases (stderr)
""", """            btr("regrow");
            double ph_feed = now_ms() - tph0, ph_open = 0, ph_tail = 0;   // the request's prompt phases (stderr)
""")
# feed: per batch
rep("""                int B = std::min(pb_cap, n - c0);
""", """                int B = std::min(pb_cap, n - c0);
                btr("batch start", B);
""")
rep("""                // larry: the next batch's Engram rows go into the row cache while this one computes
""", """                btr("columns + begin_batch");
                // larry: the next batch's Engram rows go into the row cache while this one computes
""")
rep("""                    if (pf_last >= 0 && !pf_shallow && !mc.cpu_only) tier.prefetch_begin(pf_first, pf_last, pipe ? part_max : B);
""", """                    if (pf_last >= 0 && !pf_shallow && !mc.cpu_only) tier.prefetch_begin(pf_first, pf_last, pipe ? part_max : B);
                    btr("prefetch begin");
""")
rep("""                    okp = okp && PJ();
                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
""", """                    okp = okp && PJ();
                    if (!okp) tier.run_chunk_wait();   // never leave the tier thread running
                    btr("pipeline");
""")
rep("""                tier.prefetch_end();
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
""", """                tier.prefetch_end();
                fd_dense += now_ms() - t0;
                btr("layers / prefetch end");
                const double tl = now_ms();
""")
rep("""                fd_logits += now_ms() - tl;
                c0 += B;
""", """                fd_logits += now_ms() - tl;
                btr("logits");
                c0 += B;
""")
open(P, "w").write(s)
print("patched", P)
