#!/usr/bin/env python3
# patch-ropeip.py [DENSE_CPP] - larry, STRATA_ROPE_INPLACE=1 (default 0 until measured): the prompt path's two big RoPEs -
# q's (after q_b, [512, 64 heads, n]) and the attention output's de-RoPE - in place.  DeepSeek-V4 rotates only the last
# 64 of 512 dims; out of place the op copied all 512 (67 MB a 512-column span, ~240 us each); in place ggml-sycl's rope
# (patch-glue1.py) touches only the rotated pairs.  Neither tensor is read un-rotated afterwards.  Bit-identical.
# Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_ROPE_INPLACE" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:100]))
    s = s.replace(old, new)


rep("""    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {""",
    """    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false, bool inplace = false) const {""")
rep("""        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);""",
    """        ggml_tensor * r = nullptr;
        if (inplace) {   // larry, STRATA_ROPE_INPLACE: a view of a (ggml_rope_ext_back is ggml_rope_ext with the op swapped)
            r = ggml_rope_ext_inplace(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
            if (back) r->op = GGML_OP_ROPE_BACK;
        } else {
            r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                     : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        }""")
rep("""    if (!afuse) q = b.rope(q, pos, r);
""", """    // larry, STRATA_ROPE_INPLACE (default 0): in place - ggml-sycl then rotates only the 64 rope dims (q is read rotated only)
    static const bool rope_ip = env_on("STRATA_ROPE_INPLACE", false);
    if (!afuse) q = b.rope(q, pos, r, false, rope_ip && im.sycl);
""")
rep("""    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
""", """    o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                           // de-RoPE the output
""")
open(P, "w").write(s)
print("patched", P)
