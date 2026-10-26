#!/usr/bin/env python3
# patch-dsprof.py - STRATA_DSPARK_PROF=1 (diagnostic): the draft step's host phases (inputs, graph allocation, the
# graph compute call, readbacks), averaged and printed every 64 drafts.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dspark.cpp"
s = open(P).read()
if "STRATA_DSPARK_PROF" in s: print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""    Impl::G & gr = im.graph(k, draft);""", """    static const bool dprof = std::getenv("STRATA_DSPARK_PROF") != nullptr;
    static double pt[5] = {}; static long pn = 0;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto msd = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto ti = now();
    Impl::G & gr = im.graph(k, draft);""")
rep("""    if (!ggml_gallocr_alloc_graph(im.ga, gr.gf)) { im.err = "DSpark graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(im.be, gr.gf) != GGML_STATUS_SUCCESS) { im.err = "DSpark graph compute failed"; return false; }""",
    """    if (!ggml_gallocr_alloc_graph(im.ga, gr.gf)) { im.err = "DSpark graph allocation failed"; return false; }
    const auto ta = now();
    if (ggml_backend_graph_compute_async(im.be, gr.gf) != GGML_STATUS_SUCCESS) { im.err = "DSpark graph compute failed"; return false; }
    const auto tq = now();
    ggml_backend_synchronize(im.be);
    const auto tc = now();""")
rep("""    im.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;""", """    im.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (dprof && draft) {
        pt[0] += msd(t0, ti); pt[1] += msd(ti, ta); pt[2] += msd(ta, tq); pt[3] += msd(tq, tc); pt[4] += msd(tc, now());
        if (++pn % 64 == 0)
            std::fprintf(stderr, "dspark prof (%ld drafts): inputs %.2f ms, graph/alloc %.2f, enqueue %.2f, sync %.2f, readback %.2f, total %.2f\\n",
                         pn, pt[0] / pn, pt[1] / pn, pt[2] / pn, pt[3] / pn, pt[4] / pn, (pt[0] + pt[1] + pt[2] + pt[3] + pt[4]) / pn);
    }
    return true;""")
open(P, "w").write(s)
print("patched")
