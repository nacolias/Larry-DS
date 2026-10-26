#!/usr/bin/env python3
# patch-colprof.py [DENSE_CPP] - larry, STRATA_DENSE_PROF: set_columns prints how long the state columns' reallocation
# took (a batch over S columns allocates ~0.32 MiB a column on card 0 - 5 GiB for 16,384 - and serve's elastic columns
# give it back after every batch).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "state columns in %.0f ms" in s:
    print("already patched"); sys.exit(0)
old1 = """    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    ggml_backend_synchronize(im.backend);
"""
new1 = """    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    const double tc0 = g_dprof_on ? dprof_now() : 0;
    ggml_backend_synchronize(im.backend);
"""
old2 = """    im.gctx = ggml_init(gp);
    return alloc_cols(im, want, im.err);
}
"""
new2 = """    im.gctx = ggml_init(gp);
    const bool ok = alloc_cols(im, want, im.err);
    if (g_dprof_on) std::fprintf(stderr, "dense prof: %lld state columns in %.0f ms\\n", (long long) want, dprof_now() - tc0);
    return ok;
}
"""
for o in (old1, old2):
    if s.count(o) != 1:
        sys.exit("anchor x%d: %s" % (s.count(o), o[:70]))
open(P, "w").write(s.replace(old1, new1).replace(old2, new2))
print("patched", P)
