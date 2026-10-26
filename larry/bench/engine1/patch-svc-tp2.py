#!/usr/bin/env python3
# patch-svc-tp2.py - larry: the service engine script (ds41-xe-engine.sh) runs engine1 (STRATA_TP=4) at the service's
# own expert settings (FLEX 1 GiB / margin 1.5 stay); a backup is kept.  Run on larry after the one-shot validation.
import shutil, sys, time
p = '/opt/ai/dsv41/ds41-xe-engine.sh'
s = open(p).read()
old = 'export STRATA_TP="${STRATA_TP-0}"'
if old not in s:
    print('no STRATA_TP=0 line (already patched?)'); sys.exit(1)
shutil.copy(p, p + '.pre-tp2-' + time.strftime('%Y%m%d%H%M'))
start = s.index('# larry, engine1:')
end = s.index(old) + len(old)
new = '''# larry, engine1: decode / verify passes (<= 7 tokens) on the four cards' tensor-parallel core - the dense half split by
# heads / columns over the cards, each card its own experts, a replayed SYCL graph a card a pass; its KV replicas map
# as the context grows (the flex region holds that VRAM meanwhile), promotions repack on the cards.  One-shot A/B at
# these settings (ctx 262144, FLEX 1): DSpark 60.1 -> 67.8 tok/s, verify 35.3 -> 29.5 ms/pass.  0 = the old path
export STRATA_TP="${STRATA_TP-4}"'''
s = s[:start] + new + s[end:]
fo = '''# the promotion cache (each expert card's flex region, also a prompt's prefetch buffers): 3 GiB a card (was 1) - TP's
# weights take ~2 GB a card from the experts, the slots win most of it back (both paths: fewer CPU-pool experts)'''
fn = '''# the promotion cache (each expert card's flex region, also a prompt's prefetch buffers): 1 GiB a card, 1.5 GiB kept
# free (3 GiB / 0.3 left a 252-token prompt's chunk buffers no VRAM: OUT_OF_DEVICE_MEMORY, 2026-10-23)'''
if fo in s: s = s.replace(fo, fn)
open(p, 'w').write(s)
print('patched')
