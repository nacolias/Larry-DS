#!/usr/bin/env python3
# svc-glue.py [ENGINE_SH] - larry, 2026-10-25 ~07:00: the dense span's glue kernels into the service (patch-glue1/2.py,
# patch-ropeip.py; all default off in the code): hc_post one work-item per (dim, token), q's RoPE and the attention
# output's de-RoPE in place over the 64 rotated dims only, the DSV4 MQA prompt attention's Q conversion vectorised and
# its softmax a sub-group a row.  abwT24-T27: card 0's dense part 235 -> ~220 ms a layer at 16K; fresh 16K ~9.5 ->
# ~8.6-9.0 s, 31K ~16.2-16.5 -> ~14.9-15.3 s.  Backup .pre-glue-20261025.  Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "GGML_SYCL_HC_POST_V2" in s:
    print("already patched"); sys.exit(0)
old = """exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
new = """# the dense span's glue kernels (2026-10-25, abwT24-T27; README "Measured (2026-10-25 05:30 - 07:00: ...)"): hc_post
# reads the hc state once (it read it once per output), q's RoPE and the attention output's de-RoPE in place over the
# 64 rotated dims (they copied all 512), the DSV4 MQA prompt attention's Q conversion vectorised and its softmax a
# sub-group a row (three passes; the work-group form's two group reductions a row were the cost)
export GGML_SYCL_HC_POST_V2="${GGML_SYCL_HC_POST_V2-1}"
export STRATA_ROPE_INPLACE="${STRATA_ROPE_INPLACE-1}"
export GGML_SYCL_DSV4_MQA_QV="${GGML_SYCL_DSV4_MQA_QV-1}"
export GGML_SYCL_DSV4_MQA_SM="${GGML_SYCL_DSV4_MQA_SM-1}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
shutil.copy(P, P + ".pre-glue-20261025")
open(P, "w").write(s.replace(old, new))
print("patched", P)
