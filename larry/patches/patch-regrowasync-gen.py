#!/usr/bin/env python3
# patch-regrowasync-gen.py [GENERATE_CPP] - larry, STRATA_SERVE_REGROW_ASYNC=1 (default 0; with patch-regrowasync.py):
# a kept batch's cache regrow is queued (regrow_cache_async) instead of waited for - the first token no longer waits
# for its copies (0.15-0.47 s).  TP decode only: every TP pass first publishes a regrow whose copies are done
# (tp_run -> regrow_poll).  The route bias right after the batch already counts the queued experts (resident()), so
# nothing else changes.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(P).read()
if "STRATA_SERVE_REGROW_ASYNC" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    auto tp_run = [&](const int * toks, int m, int pos, float * vh_out, float * lg_out = nullptr) -> bool {
        int64_t lo = 0, hi = 0;
""", """    auto tp_run = [&](const int * toks, int m, int pos, float * vh_out, float * lg_out = nullptr) -> bool {
#if defined(DS41_XE)
        tier.regrow_poll(false);   // larry, STRATA_SERVE_REGROW_ASYNC: a queued regrow whose copies are done - in this pass's table
#endif
        int64_t lo = 0, hi = 0;
""")
rep("""                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
                if (tier.regrow_cache(kerr) < 0) {""", """                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
#if defined(DS41_XE)
                // larry, STRATA_SERVE_REGROW_ASYNC (patch-regrowasync-gen.py): the copies left running (TP decode publishes them)
                static const bool rg_async = tp_on && std::getenv("STRATA_SERVE_REGROW_ASYNC") && std::atoi(std::getenv("STRATA_SERVE_REGROW_ASYNC")) != 0;
                if ((rg_async ? tier.regrow_cache_async(kerr) : tier.regrow_cache(kerr)) < 0) {
#else
                if (tier.regrow_cache(kerr) < 0) {
#endif""")
open(P, "w").write(s)
print("patched", P)
