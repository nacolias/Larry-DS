#!/usr/bin/env python3
# patch-engasync.py [DENSE_CPP] - larry, STRATA_ENG_START_ASYNC=1 (default 0 until measured): a prompt batch's Engram
# start (hash every column's n-grams, sort a slot's row ids, look them up in the row cache, plan the reads: 0.08-0.15 s
# on the caller for 8-16K columns, STRATA_DENSE_PROF) on a thread, so layer 0's spans run meanwhile - the rows are first
# needed at layer 1.  The batch is "pending" while the thread runs: the waits resolve it first, and a next-batch
# prefetch is not launched under it (eng_pf_launch rewrites the token history start_into reads).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_ENG_START_ASYNC" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


if "#include <future>" not in s:
    rep("#include <map>\n", "#include <future>\n#include <map>\n")
rep("""    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
""", """    std::shared_ptr<Ds41Engram::Batch> eng_pf;   // larry: the next batch's rows being read into the row cache
    // larry, STRATA_ENG_START_ASYNC: a prompt batch's start_into on a thread (eng_batch is set when it is resolved)
    std::future<std::shared_ptr<Ds41Engram::Batch>> eng_fut;
    std::string eng_fut_err;
""")
# resolve helper before engram_wait_host
rep("""static bool engram_wait_host(Ds41Dense::Impl & im, int slot) {
    if (!im.eng_batch""", """// larry, STRATA_ENG_START_ASYNC: the started batch from its thread (a no-op without one)
static bool eng_resolve(Ds41Dense::Impl & im) {
    if (!im.eng_fut.valid()) return true;
    im.eng_batch = im.eng_fut.get();
    if (!im.eng_batch) { im.err = im.eng_fut_err.empty() ? "engram: batch start failed" : im.eng_fut_err; im.eng_pin_n = 0; return false; }
    return true;
}

static bool engram_wait_host(Ds41Dense::Impl & im, int slot) {
    if (!eng_resolve(im)) return false;
    if (!im.eng_batch""")
rep("""static bool engram_ready(Ds41Dense::Impl & im, int slot) {
    if (!im.eng_batch) return true;""", """static bool engram_ready(Ds41Dense::Impl & im, int slot) {
    if (!eng_resolve(im)) return false;
    if (!im.eng_batch) return true;""")
rep("""            if (eng_pin_ok(im, per)) {
                im.eng_batch = im.engram.start_into(pos0, n, im.eng_pin, im.err);
                im.eng_pin_n = im.eng_batch ? per : 0;
            } else {""", """            static const bool start_async = env_on("STRATA_ENG_START_ASYNC", false);
            if (start_async && eng_pin_ok(im, per)) {   // larry: the start on a thread, resolved at the first wait
                im.eng_pin_n = per;
                im.eng_fut_err.clear();
                float * const * outp = im.eng_pin;
                im.eng_fut = std::async(std::launch::async, [&im, pos0, n, outp]() {
                    return im.engram.start_into(pos0, n, outp, im.eng_fut_err);
                });
            } else if (eng_pin_ok(im, per)) {
                im.eng_batch = im.engram.start_into(pos0, n, im.eng_pin, im.err);
                im.eng_pin_n = im.eng_batch ? per : 0;
            } else {""")
rep("""            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch) return false;
        }
    }""", """            for (int e = 0; e < 2; ++e) im.eng_waited[e] = im.eng_up[e] = false;
            if (!im.eng_batch && !im.eng_fut.valid()) return false;
        }
    }""")
rep("""    im.eng_pf_pos0 = pos0;
    if (!im.eng_batch) eng_pf_launch(im);""", """    im.eng_pf_pos0 = pos0;
    if (!im.eng_batch && !im.eng_fut.valid()) eng_pf_launch(im);   // larry: not under a batch still starting""")
open(P, "w").write(s)
print("patched", P)
