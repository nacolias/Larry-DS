#!/usr/bin/env python3
# patch-woa3.py - fix: larry_slices_gemm's lazy fallback W buffer must be the pool's TOP allocation.  ggml-sycl's VMM
# pool frees in stack order (ggml-sycl.cpp:1896 asserts it): wf declared before xf but allocated after it was freed
# second -> "GGML_ASSERT(ptr == pool_addr + pool_used)" whenever the fp16 weight cache had no room (the service:
# a 34-token prompt).  xf first, wf second: wf (if allocated) is on top and freed first.  Idempotent.
import sys
S = "/opt/ai/dsv41/xolotl/third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp"
s = open(S).read()
old = """    ggml_sycl_pool_alloc<sycl::half> wf(ctx.pool());   // only when the fp16 weight cache cannot hold the slice
    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);
"""
new = """    ggml_sycl_pool_alloc<sycl::half> xf(ctx.pool(), (size_t) ncols * K);
    // only when the fp16 weight cache cannot hold the slice; declared after xf: the pool frees in stack order
    ggml_sycl_pool_alloc<sycl::half> wf(ctx.pool());
"""
if new in s: print("already"); sys.exit(0)
if s.count(old) != 1: sys.exit("anchor")
open(S, "w").write(s.replace(old, new))
print("patched")
