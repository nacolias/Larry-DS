#!/usr/bin/env python3
# patch-svcenv.py [qkv] - the service engine script (ds41-xe-engine.sh): the 2026-10-22 measured wins - the decode /
# verify lightning indexer on XMX (GGML_SYCL_LI_GEMM_MIN=1, GGML_SYCL_LI_XMX=1), STRATA_XE_JOBNT=1, STRATA_XE_COMBINE=1,
# and STRATA_QKV_FUSE=1 when called with "qkv".  Backup .bak-20261022.  Idempotent.
import sys, shutil
P = "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "GGML_SYCL_LI_XMX" in s:
    print("already"); sys.exit(0)
shutil.copy(P, P + ".bak-20261022")
add = """# the lightning indexer of decode / verify passes (<= 8 queries) on XMX: the per-row kernel took ~23 ns a key row a
# query (98K context ~2.3 ms an indexer layer per query, 8 indexer layers); a 95K-context DSpark essay 21.6 -> 53-55
# tok/s with both (short context and decode-path PPL unchanged)
export GGML_SYCL_LI_GEMM_MIN="${GGML_SYCL_LI_GEMM_MIN:-1}"
export GGML_SYCL_LI_XMX="${GGML_SYCL_LI_XMX:-1}"
# a decode / verify job's own token count picks its expert kernel form (1-token jobs of verify passes 9-17 % faster)
export STRATA_XE_JOBNT="${STRATA_XE_JOBNT:-1}"
# prompt chunks: each expert card pushes its part to the dense card when it is done (31.6K prompt 23.0 -> 21.5 s, exact)
export STRATA_XE_COMBINE="${STRATA_XE_COMBINE:-1}"
"""
if len(sys.argv) > 1 and sys.argv[1] == "qkv":
    add += """# q_a + kv as one weight / one product (QKV_FUSE)
export STRATA_QKV_FUSE="${STRATA_QKV_FUSE:-1}"
"""
anchor = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"\n'
assert s.count(anchor) == 1
s = s.replace(anchor, add + anchor)
open(P, "w").write(s)
print("patched")
