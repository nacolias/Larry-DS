#!/usr/bin/env python3
# patch-fnbuf.py [DENSE_CPP] - larry: o_fn (the FFN input [D, columns], which the expert cards' copy engines read
# card-to-card in a prompt batch) in its own buffer, allocated once for the whole prompt batch (NCmax columns, 335 MB
# at 16,384) instead of inside the elastic state columns: serve gives the columns back after every batch, so each fresh
# batch's first card-to-card read of a NEW ~5 GiB state buffer paid its peer mapping (the tier's first X copy blocked
# ~350-500 ms, under the tier's submit lock: the first big prompts after a load waited 2.5-7.8 s for that lock).
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "fn_buf" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:90]))
    s = s.replace(old, new)


rep("""    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
""", """    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    // larry: o_fn's own buffer, NCmax columns for the engine's life (the expert cards read it card-to-card; a fresh
    // buffer per batch paid the peer mapping at the tier's first copy)
    ggml_context * fn_ctx = nullptr;
    ggml_backend_buffer_t fn_buf = nullptr;
""")
rep("""    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
""", """    if (!im.fn_buf) {   // larry: allocated once, NCmax columns (views take the first NC)
        ggml_init_params fp = { ggml_tensor_overhead() * 2, nullptr, true };
        im.fn_ctx = ggml_init(fp);
        ggml_tensor * fn = ggml_new_tensor_2d(im.fn_ctx, GGML_TYPE_F32, D, std::max(NC, im.NCmax));
        im.fn_buf = ggml_backend_alloc_ctx_tensors(im.fn_ctx, im.backend);
        if (!im.fn_buf) { err = "cannot allocate the FFN input columns"; return false; }
        ggml_backend_buffer_clear(im.fn_buf, 0);
        im.o_fn = fn;
    }
""")
open(P, "w").write(s)
print("patched", P)
