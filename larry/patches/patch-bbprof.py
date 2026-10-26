#!/usr/bin/env python3
# patch-bbprof.py [DENSE_CPP] - larry, STRATA_DENSE_PROF: begin_batch's parts (a 16K batch's columns + begin_batch took
# 0.4-0.86 s, STRATA_BATCH_TRACE): the host embeddings, the input uploads, the Engram start, the init graphs.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "dense prof: begin_batch" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    // embeddings (F16 rows on the mmap)
    im.host_emb.resize((size_t) (D * n));""", """    // embeddings (F16 rows on the mmap)
    const double tu0 = g_dprof_on && n > 64 ? dprof_now() : 0;   // larry, STRATA_DENSE_PROF: the batch inputs' parts
    im.host_emb.resize((size_t) (D * n));""")
rep("""    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);
    std::vector<int32_t> pos(n);""", """    const double tu1 = g_dprof_on && n > 64 ? dprof_now() : 0;
    im.up_set(im.i_emb, im.host_emb.data(), im.host_emb.size() * 4);
    const double tu2 = g_dprof_on && n > 64 ? dprof_now() : 0;
    std::vector<int32_t> pos(n);""")
rep("""    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows""",
    """    const double tu3 = g_dprof_on && n > 64 ? dprof_now() : 0;
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows""")
rep("""    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for it
    return true;
}""", """    if (im.up_async) ggml_backend_event_record(im.up_ev, im.backend);   // the staging buffer's reuse waits for it
    if (g_dprof_on && n > 64)
        std::fprintf(stderr, "dense prof: begin_batch inputs (%d tokens): embeddings %.1f ms, their upload %.1f ms, maps %.1f ms, "
                     "Engram start %.1f ms\\n", n, tu1 - tu0, tu2 - tu1, tu3 - tu2, dprof_now() - tu3);
    return true;
}""")
rep("""    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));""",
    """    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    const double ti0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));""")
rep("""        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}""", """        if (!ok) { im.err = "batch init failed"; return false; }
    }
    if (g_dprof_on) std::fprintf(stderr, "dense prof: begin_batch init graphs (%lld spans) %.1f ms\\n", (long long) nb, dprof_now() - ti0);
    return true;
}""")
open(P, "w").write(s)
print("patched", P)
