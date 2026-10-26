#!/usr/bin/env python3
# svc-denselite-cmt.py - larry: ds41-xe-engine.sh's DENSE_LITE comment after patch-litemin0 (no streamed experts on card 0
# in every pipelined call; STRATA_XE_DENSE_LITE_MIN default 0).  Idempotent.
P = "/opt/ai/dsv41/ds41-xe-engine.sh"
s = open(P).read()
old = """# ~90 % of the lower layers' calls (STRATA_XE_CALL_TL timelines).  DENSE_LITE: X read in place, no host waits, and in
# calls of >= 2048 tokens (STRATA_XE_DENSE_LITE_MIN) no streamed experts on card 0; DENSE_INLINE:"""
new = """# ~90 % of the lower layers' calls (STRATA_XE_CALL_TL timelines).  DENSE_LITE: X read in place, no host waits, and no
# streamed experts on card 0 (in calls of >= STRATA_XE_DENSE_LITE_MIN tokens, default 0 = all); DENSE_INLINE:"""
if new in s:
    print("already patched")
else:
    assert s.count(old) == 1, "anchor"
    open(P, "w").write(s.replace(old, new))
    print("patched", P)
