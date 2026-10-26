#!/usr/bin/env python3
# patch-svc-tp.py - larry: the service engine script (ds41-xe-engine.sh) gets engine1's settings (run on larry, after
# the one-shot validation; a backup is kept)
import shutil, sys
p = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(p).read()
if 'STRATA_TP=' in s:
    print('already patched'); sys.exit(0)
shutil.copy(p, p + '.pre-tp')
old = 'exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"'
assert s.count(old) == 1
new = '''# larry, engine1: decode / verify passes (<= 7 tokens) on the four cards' tensor-parallel core - the dense half split by
# heads / columns over the cards, each card its own experts, a replayed SYCL graph a card a pass (one-shot essay A/B:
# DSpark verify 33.3 -> 29.3 ms/pass, plain decode 46.8 -> ~54 tok/s); the Q6_K weights cached on disk
export STRATA_TP="${STRATA_TP-4}"
export STRATA_TP_CACHE="${STRATA_TP_CACHE-/srv/dsv41/tpcache}"
[ "$STRATA_TP" = "0" ] && unset STRATA_TP
# the promotion cache (each expert card's flex region, also a prompt's prefetch buffers): 3 GiB a card (was 1) - TP's
# weights take ~2 GB a card from the experts, the slots win most of it back (both paths: fewer CPU-pool experts)
export STRATA_XE_FLEX_GIB="${STRATA_XE_FLEX_GIB-3}"
export STRATA_XE_FLEX_MARGIN_GIB="${STRATA_XE_FLEX_MARGIN_GIB-0.3}"
''' + old
s = s.replace(old, new)
open(p, 'w').write(s)
print('patched')
