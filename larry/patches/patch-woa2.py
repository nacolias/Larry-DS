#!/usr/bin/env python3
# patch-woa2.py - STRATA_PREREORDER_SLICES (default 1; 0 = wo_a reordered by the first decode-shaped product, as before)
import sys
E = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_generate.cpp"
s = open(E).read()
if "STRATA_PREREORDER_SLICES" in s: print("already"); sys.exit(0)
old = """        // wo_a per group slice: every prompt span then takes the per-slice GEMM, from the first prompt on
        int ns = 0;
        for (ggml_tensor * t : dense.slice_reorder_weights())
            ns += ggml_backend_sycl_prereorder_slices(dense.backend(), t, dense.geom().o_groups) ? 1 : 0;
"""
new = """        // wo_a per group slice: every prompt span then takes the per-slice GEMM, from the first prompt on
        // (STRATA_PREREORDER_SLICES, default 1)
        int ns = 0;
        if (!(std::getenv("STRATA_PREREORDER_SLICES") && std::getenv("STRATA_PREREORDER_SLICES")[0] == '0'))
            for (ggml_tensor * t : dense.slice_reorder_weights())
                ns += ggml_backend_sycl_prereorder_slices(dense.backend(), t, dense.geom().o_groups) ? 1 : 0;
"""
if s.count(old) != 1: sys.exit("anchor")
open(E, "w").write(s.replace(old, new))
print("patched")
