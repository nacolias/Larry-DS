#!/usr/bin/env python3
# svc-denselite.py [ENGINE_SH] - larry, 2026-10-25 ~04:45: the prompt path's dense-card changes into the service
# (patch-calltl / -denselite / -denseinline / -denseoff / -seedtail, all default off in the code): in a pipelined
# prompt call the dense card no longer runs the tier's streamed or cached experts beside the dense half (its part was
# the last card of ~90 % of the lower layers' calls: STRATA_XE_CALL_TL), the combine goes in the dense half's stream,
# the expert cards stream their overflow while they compute (OVERLAP), shallow batches take the prefetch too, and card
# 0's cache is laid out with layers 20-39 first so a batch's shrink gives back the lower layers' experts.  abwT12-T20:
# fresh prompts 8K ~7.1 -> ~6.7 s, 16K ~10.35 -> ~9.6 s, 31K ~17.5 -> ~16.3 s, the agent replay 118.6 -> ~115 s.
# Backup .pre-denselite-20261025.
# Idempotent.
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
if "STRATA_XE_DENSE_LITE" in s:
    print("already patched"); sys.exit(0)
old1 = """# a prompt batch that ends before context position 8192 takes no prefetch: its routed arena experts stream on demand
# and the CPU pool runs the light ones (CPU_LIGHT) - shallow batches are bound by their experts (2026-10-24 A/B: see
# the README)
export STRATA_XE_PREFETCH_DEPTH="${STRATA_XE_PREFETCH_DEPTH-8192}"
"""
new1 = """# a prompt batch that ends before context position 5000 takes no prefetch: its routed arena experts stream on demand
# and the CPU pool runs the light ones (CPU_LIGHT).  (8192 until 2026-10-25: with the dense card out of the tier's
# calls below, 6-8K batches from empty gain from the prefetch - 8K 7.24 / 7.02 -> 6.86 / 6.71 s - and 4K ones do not,
# abwT18 / abwT19)
export STRATA_XE_PREFETCH_DEPTH="${STRATA_XE_PREFETCH_DEPTH-5000}"
"""
old2 = """exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
new2 = """# prompt calls (2026-10-25, abwT12-T20; README "Measured (2026-10-25 02:45 - 05:15: the dense card out of the prompt
# calls)"): in a pipelined prompt call card 0's part of the tier ran beside its own dense half and was the last card of
# ~90 % of the lower layers' calls (STRATA_XE_CALL_TL timelines).  DENSE_LITE: X read in place, no host waits, and in
# calls of >= 2048 tokens (STRATA_XE_DENSE_LITE_MIN) no streamed experts on card 0; DENSE_INLINE:
# its part and the combine in the dense half's stream; DENSE_OFFLOAD: in calls of >= 4096 tokens card 0's cached experts
# are computed by the expert cards from their pinned arena copies; OVERLAP: the expert cards stream their overflow
# while they compute; GATHER_INIT: no memset of a card's part; SEED_TAIL_LAYERS: card 0's cache laid out with layers
# 20-39 first, so a batch's shrink gives back lower-layer experts (the upper layers are bound by the prefetch's staging)
export STRATA_XE_DENSE_LITE="${STRATA_XE_DENSE_LITE-1}"
export STRATA_XE_DENSE_INLINE="${STRATA_XE_DENSE_INLINE-1}"
export STRATA_XE_DENSE_OFFLOAD="${STRATA_XE_DENSE_OFFLOAD-4096}"
export STRATA_XE_OVERLAP="${STRATA_XE_OVERLAP-1}"
export STRATA_XE_GATHER_INIT="${STRATA_XE_GATHER_INIT-1}"
export STRATA_XE_SEED_TAIL_LAYERS="${STRATA_XE_SEED_TAIL_LAYERS-20}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
"""
for o in (old1, old2):
    if s.count(o) != 1:
        sys.exit("anchor x%d: %s" % (s.count(o), o[:70]))
shutil.copy(P, P + ".pre-denselite-20261025")
open(P, "w").write(s.replace(old1, new1).replace(old2, new2))
print("patched", P)
