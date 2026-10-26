#!/usr/bin/env python3
# patch-conts.py - larry, STRATA_CONT_SKIP (default 1): the ggml_cont before the fused fake-quant (window K, the
# indexer's q / k, the compressed latent - in the attention and the enc-skip KV graphs) and of the compressed rows'
# mask, skipped when the tensor already is contiguous - ggml_cont always adds an op (a full copy: ggml-sycl runs a
# memcpy command), and these inputs are rope outputs / whole-row views.  unitrace, plain decode: two such copies a
# layer on card 0 (~1.3 us each plus its dependency gap).  Same values (a copy dropped).
# Usage: patch-conts.py [ds41_dense.cpp]   Idempotent.
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_CONT_SKIP" in s:
    print("already"); sys.exit(0)


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt: sys.exit(f"anchor x{s.count(old)}: {old[:90]!r}")
    s = s.replace(old, new)


rep("""static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
""", """// larry, STRATA_CONT_SKIP (default 1): a ggml_cont only where the tensor is not contiguous already (ggml_cont is
// always an op - a copy - even of a contiguous tensor)
static ggml_tensor * cont_if(ggml_context * gc, ggml_tensor * x) {
    static const bool skip = env_on("STRATA_CONT_SKIP", true);
    return skip && ggml_is_contiguous(x) ? x : ggml_cont(gc, x);
}

static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
""")
rep("if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);",
    "if (im.cfg.act_quant) kv2 = fake_quant(gc, cont_if(gc, kv2), Ds41Quant::FP8_POW2_32);", 2)
rep("if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);",
    "if (im.cfg.act_quant) ik = fake_quant(gc, cont_if(gc, ik), Ds41Quant::FP4_POW2_32);", 2)
rep("if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);",
    "if (im.cfg.act_quant) lr = fake_quant(gc, cont_if(gc, lr), Ds41Quant::FP4_E4M3_16);", 2)
rep("if (im.cfg.act_quant) iq = fake_quant(gc, ggml_cont(gc, iq), Ds41Quant::FP4_POW2_32);",
    "if (im.cfg.act_quant) iq = fake_quant(gc, cont_if(gc, iq), Ds41Quant::FP4_POW2_32);")
rep("            Mc = ggml_cont(gc, selm);\n", "            Mc = cont_if(gc, selm);\n")
open(P, "w").write(s)
print("patched")
