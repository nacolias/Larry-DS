#!/usr/bin/env python3
# patch-litemin0.py [TIER_CPP] - larry: STRATA_XE_DENSE_LITE_MIN's default 2048 -> 0 (DENSE_LITE keeps streamed experts
# off the dense card in every pipelined call): letting card 0 stream in the short calls made the 1.5-2.2K-token turns
# slower, not faster (abwT20: +0.2-0.5 s each - its streams there run beside the dense half's spans again).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl/tools/v41/engine/xe/ds41_xe_tier.cpp"
s = open(P).read()
old = 'static const int lite_min = env_i("STRATA_XE_DENSE_LITE_MIN", 2048);'
new = 'static const int lite_min = env_i("STRATA_XE_DENSE_LITE_MIN", 0);   // larry, patch-litemin0 (was 2048)'
if new in s:
    print("already patched"); sys.exit(0)
if s.count(old) != 1:
    sys.exit("anchor x%d" % s.count(old))
open(P, "w").write(s.replace(old, new))
print("patched", P)
