#!/usr/bin/env python3
# patch-opprof3.py [GGML_SYCL_DIR] - larry, GGML_SYCL_OP_PROF (after patch-opprof.py / -2): larry_opprof::mark callable
# from the other ggml-sycl sources (declared in common.hpp), and the DSV41 SDPA op split by marks of its own: the Q'
# conversion, the mask, oneDNN's execute, the output conversion (the node's own interval).  Idempotent.
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl"
P, C, F = D + "/ggml-sycl.cpp", D + "/common.hpp", D + "/fattn-dsv4.cpp"
s, c, f = open(P).read(), open(C).read(), open(F).read()
if "patch-opprof3" in f:
    print("already patched"); sys.exit(0)


def rep(t, old, new, what, n=1):
    if t.count(old) != n:
        sys.exit("%s anchor x%d (want %d): %s" % (what, t.count(old), n, old[:100]))
    return t.replace(old, new)


s = rep(s, "static bool on() { static const bool v = std::getenv(\"GGML_SYCL_OP_PROF\") != nullptr; return v; }",
        "bool on() { static const bool v = std::getenv(\"GGML_SYCL_OP_PROF\") != nullptr; return v; }   // (extern: patch-opprof3)", "cpp")
s = rep(s, "static void mark(sycl::queue & q, int device, const char * label) {",
        "void mark(sycl::queue & q, int device, const char * label) {   // (extern: patch-opprof3)", "cpp")
c = rep(c, "#if GGML_SYCL_DNNL\n    dnnl::engine make_engine(sycl::queue* q) {",
        "#if GGML_SYCL_DNNL\n    dnnl::engine make_engine(sycl::queue* q) {", "hpp")
# the declarations, once, at the top of common.hpp after its includes: find the first 'namespace' or struct after the includes
k = c.find("\nstruct ggml_backend_sycl_context")
if k < 0:
    sys.exit("hpp: ggml_backend_sycl_context not found")
c = c[:k] + ("\n// larry, GGML_SYCL_OP_PROF (patch-opprof3.py): ggml-sycl.cpp's op profiler, a mark from any source\n"
             "namespace larry_opprof {\nbool on();\nvoid mark(sycl::queue & q, int device, const char * label);\n}\n") + c[k:]
f = rep(f, """            dnnl::graph::tensor tout(P->out, eng, Op.get());
            P->cp.execute(strm, tin, {tout});""", """            dnnl::graph::tensor tout(P->out, eng, Op.get());
            if (larry_opprof::on()) larry_opprof::mark(*stream, ctx.device, "SDPA: Q' + mask kernels, tensors");   // patch-opprof3
            P->cp.execute(strm, tin, {tout});
            if (larry_opprof::on()) larry_opprof::mark(*stream, ctx.device, "SDPA: oneDNN execute");""", "fattn")
open(P, "w").write(s)
open(C, "w").write(c)
open(F, "w").write(f)
print("patched", P, C, F)
