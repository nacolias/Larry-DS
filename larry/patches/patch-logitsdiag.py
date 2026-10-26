#!/usr/bin/env python3
# patch-logitsdiag.py [DENSE_CPP] - larry (diagnostic, after patch-ptl.py): a slow batch head (abwT64: two of four fresh
# 8K prompts' single-row logits took 0.79 / 0.87 s with every card idle - STRATA_GPU_PTL showed nothing running) split
# into the graph build, its allocation, the queueing (ggml-sycl's submission, the LM head split's hook included) and
# the wait; with STRATA_GPU_PTL a stamp before (50) and after (51) the graph on the dense queue (under the submit lock
# when the pipeline set one) shows whether the card or the host took the time.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "patch-logitsdiag" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    bool compute(ggml_cgraph * gf, bool cached, bool async = false, bool nosync = false);
""", """    bool compute(ggml_cgraph * gf, bool cached, bool async = false, bool nosync = false);
    double last_alloc_ms = 0;   // larry, patch-logitsdiag: the last compute()'s graph allocation
""")
rep("""    bool ok_alloc = a.alloc(gf, reuse);
    const double tq1 = g_dprof_on ? dprof_now() : 0;
""", """    const double tqa = dprof_now();   // larry, patch-logitsdiag
    bool ok_alloc = a.alloc(gf, reuse);
    last_alloc_ms = dprof_now() - tqa;
    const double tq1 = g_dprof_on ? dprof_now() : 0;
""")
rep("""    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    const bool ok = im.compute(gf, false);
    const double tl2 = dprof_now();
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    const double tl3 = dprof_now();
    if (tl3 - tl0 > 300) {
        size_t fr = 0, tot = 0;
        if (im.backend) ggml_backend_dev_memory(ggml_backend_get_device(im.backend), &fr, &tot);
        std::fprintf(stderr, "ds41_dense: slow logits (%d rows): flush %.0f ms, graph + compute %.0f ms, read %.0f ms; %.2f GiB free\\n",
                     nrows, tl1 - tl0, tl2 - tl1, tl3 - tl2, fr / 1073741824.0);
    }
    return ok;
}
""", """    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    // larry, patch-logitsdiag: queued (allocation inside), stamped, then waited for - the same work as compute(gf, false)
    const double tlb = dprof_now();
    auto stamp = [&](int tag, double t) {
        if (!ds41_dense_ptl_hook) return;
        std::unique_lock<std::mutex> sl;
        if (im.sub_mu) sl = std::unique_lock<std::mutex>(*im.sub_mu);
        ds41_dense_ptl_hook(tag, -1, nrows, t);
    };
    stamp(50, tlb);
    const bool ok = im.compute(gf, false, false, true);
    const double tlq = dprof_now();
    stamp(51, tlq);
    if (ok) ggml_backend_synchronize(im.backend);
    const double tl2 = dprof_now();
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    const double tl3 = dprof_now();
    if (tl3 - tl0 > 300) {
        size_t fr = 0, tot = 0;
        if (im.backend) ggml_backend_dev_memory(ggml_backend_get_device(im.backend), &fr, &tot);
        std::fprintf(stderr, "ds41_dense: slow logits (%d rows): flush %.0f ms, build %.0f, alloc %.0f, queue %.0f, wait %.0f, read %.0f ms; "
                     "%.2f GiB free\\n", nrows, tl1 - tl0, tlb - tl1, im.last_alloc_ms, tlq - tlb - im.last_alloc_ms, tl2 - tlq, tl3 - tl2,
                     fr / 1073741824.0);
    }
    return ok;
}
""")
open(P, "w").write(s)
print("patched", P)
