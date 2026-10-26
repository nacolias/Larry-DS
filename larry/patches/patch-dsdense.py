#!/usr/bin/env python3
# patch-dsdense.py - larry: STRATA_DSPARK_DENSE_TYPE=<ggml type> (q6_K): the draft stack's Q8_0 projections (attn,
# shared expert, main_proj, Markov head) requantized ON LOAD like the experts, so ggml-sycl runs them on fastmv's
# Q6_K kernels (one launch for wo_a's 8 groups instead of 8 generic Q8_0 mat-vecs at ~35 GB/s; ~1 ms a draft).
# The router (ffn.gate), the embeddings (get_rows) and every non-Q8_0 tensor keep their type.  Idempotent.
import sys
P = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dspark.cpp"
s = open(P).read()
if "STRATA_DSPARK_DENSE_TYPE" in s: print("already"); sys.exit(0)
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor x{s.count(old)}: {old[:80]!r}")
    s = s.replace(old, new)
rep("""    auto is_exp = [](const std::string & n) { return n.find(".ffn.experts.") != std::string::npos; };""",
    """    auto is_exp = [](const std::string & n) { return n.find(".ffn.experts.") != std::string::npos; };
    // larry, STRATA_DSPARK_DENSE_TYPE=<ggml type> (q6_K): the draft's Q8_0 projections requantized on load as well
    ggml_type dt = GGML_TYPE_COUNT;
    if (const char * x = std::getenv("STRATA_DSPARK_DENSE_TYPE"); x && *x) {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * nm = ggml_type_name((ggml_type) i);
            if (nm && std::strcmp(nm, x) == 0) dt = (ggml_type) i;
        }
        if (dt == GGML_TYPE_COUNT) { e = std::string("STRATA_DSPARK_DENSE_TYPE: unknown type ") + x; return false; }
        if (ggml_quantize_requires_imatrix(dt)) { e = std::string("STRATA_DSPARK_DENSE_TYPE ") + x + " needs an imatrix"; return false; }
        ggml_quantize_init(dt);
    }
    auto is_dense_rq = [&](const std::string & n, ggml_type st) {   // a Q8_0 projection matrix (not the router / embeddings)
        if (dt == GGML_TYPE_COUNT || st != GGML_TYPE_Q8_0 || is_exp(n)) return false;
        auto ends = [&](const char * x) { const size_t l = std::strlen(x); return n.size() > l && n.compare(n.size() - l, l, x) == 0; };
        return !ends(".ffn.gate.weight") && !ends("markov_head.embed.weight");
    };""")
rep("""        const ggml_type ty = (xt != GGML_TYPE_COUNT && is_exp(s.second->name)) ? xt : (ggml_type) s.second->type;""",
    """        const ggml_type ty = (xt != GGML_TYPE_COUNT && is_exp(s.second->name)) ? xt
                           : is_dense_rq(s.second->name, (ggml_type) s.second->type) ? dt : (ggml_type) s.second->type;""")
open(P, "w").write(s)
print("patched")
