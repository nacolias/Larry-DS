#!/usr/bin/env python3
# patch-sdpa-dense.py [DENSE_CPP] - larry, STRATA_ATTN_SDPA=1 (default 0 until measured): the prompt attention of a span
# whose key count is fixed (the window-only layers: 128; the top-k layers: 128 + 512) as ggml_dsv41_kgather_pad +
# ggml_dsv41_attn_sdpa (patch-sdpa.py: one oneDNN graph SDPA a full span, rows of 576 halves with the sink key) in
# place of KGATHER + the oneMKL MQA path.  The upper layers (all of a context's compressed rows: their count moves)
# keep the old path.  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(P).read()
if "STRATA_ATTN_SDPA" in s:
    print("already patched"); sys.exit(0)
old = """    if (!im.cpu && !sycl_mqa && !sycl_dec && M % 256) {"""
new = """    // larry, STRATA_ATTN_SDPA (patch-sdpa-dense.py): a fixed key count's span as one oneDNN SDPA (keys 576 wide + the sink)
    static const bool attn_sdpa = env_on("STRATA_ATTN_SDPA", false);
    static const int sdpa_dp = std::getenv("STRATA_ATTN_SDPA_DP") ? std::atoi(std::getenv("STRATA_ATTN_SDPA_DP")) : 576;
    if (attn_sdpa && sycl_mqa && DH == 512 && (M == SWA || (Kc && M == SWA + 512))) {
        ggml_tensor * kp = ggml_dsv41_kgather_pad(gc, wsrc, widx_v, Kc ? kg_comp : nullptr, Kc ? kg_sel : nullptr, (int) SWA, sdpa_dp);
        o = ggml_dsv41_attn_sdpa(gc, q, kp, Mall, im.BL(il, "attn_sinks.weight"), 1.0f / std::sqrt((float) DH));   // [DH, NH, n]
        o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                       // de-RoPE the output
    } else {
    if (!im.cpu && !sycl_mqa && !sycl_dec && M % 256) {"""
if s.count(old) != 1:
    sys.exit("anchor 1 x%d" % s.count(old))
s = s.replace(old, new)
old2 = """    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                           // de-RoPE the output
    }"""
new2 = """    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true, rope_ip && im.sycl);                                           // de-RoPE the output
    }   // larry, STRATA_ATTN_SDPA's else
    }"""
if s.count(old2) != 1:
    sys.exit("anchor 2 x%d" % s.count(old2))
s = s.replace(old2, new2)
open(P, "w").write(s)
print("patched", P)
