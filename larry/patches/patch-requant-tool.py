#!/usr/bin/env python3
# patch-requant-tool.py [XOLOTL_DIR] - larry, 2026-10-29 (optimization #4): the ds41_requant tool (xe/ds41_requant.cpp,
# copied from patches/ds41_requant.cpp) and its target in xe.cmake (CPU only: ds41_moe_cpu, strata_kernels_cpu, ggml).
# Idempotent.
import os
import shutil
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
here = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(X, "tools/v41/engine/xe")
shutil.copyfile(os.path.join(here, "ds41_requant.cpp"), os.path.join(E, "ds41_requant.cpp"))
C = os.path.join(E, "xe.cmake")
s = open(C).read()
if "ds41_requant" in s:
    print("cmake: already has the target"); sys.exit(0)
old = "    add_executable(bench_cpu_pool ${_xe_dir}/bench_cpu_pool.cpp)\n"
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
s = s.replace(old, """    # larry (patch-requant-tool.py): the coldest experts' Q3_K parts requantized into a sidecar (STRATA_BLOB_RQ)
    add_executable(ds41_requant ${_xe_dir}/ds41_requant.cpp)
    target_link_libraries(ds41_requant PRIVATE ds41_moe_cpu strata_kernels_cpu ggml-cpu ggml-base pthread)
""" + old)
open(C, "w").write(s)
print("patched", C)
