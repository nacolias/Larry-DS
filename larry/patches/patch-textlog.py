#!/usr/bin/env python3
# patch-textlog.py [XOLOTL_DIR] - larry, 2026-10-27 (concurrent sessions): serve_ds41.py's full text log (STRATA_TEXT_LOG)
# per request - with two requests answered at once their text interleaves in the file, so every section line and the
# request's header / end carry its number (#n), and the section label is the request's own (thread-local), not shared.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
P = os.path.join(X, "tools/v41/serve/serve_ds41.py")
s = open(P).read()
if "_text_tls" in s:
    print("already patched", P)
    sys.exit(0)
pairs = [
('''_text_lock = _th.Lock()
_text_state = {"kind": None}
''',
'''_text_lock = _th.Lock()
_text_tls = _th.local()   # larry (patch-textlog.py): each request's own section label and number (#n)
_text_seq = [0]
'''),
('''        label = _KIND_LABEL.get(ev.kind, ev.kind)
        if label != _text_state["kind"]:
            _text_state["kind"] = label
            out.append(f"\\n--- {label} ---\\n")
''',
'''        label = _KIND_LABEL.get(ev.kind, ev.kind)
        if label != getattr(_text_tls, "kind", None):
            _text_tls.kind = label
            out.append(f"\\n--- #{getattr(_text_tls, 'rid', 0)} {label} ---\\n")
'''),
('''    t0 = _time.time()
    _text_state["kind"] = None
    _text_write(f"\\n\\n===== {_time.strftime('%Y-%m-%d %H:%M:%S')}  request: prompt {len(ids):,} tokens, max {max_new}, "
''',
'''    t0 = _time.time()
    with _text_lock:
        _text_seq[0] += 1
        rid = _text_seq[0]
    _text_tls.kind, _text_tls.rid = None, rid
    _text_write(f"\\n\\n===== #{rid} {_time.strftime('%Y-%m-%d %H:%M:%S')}  request: prompt {len(ids):,} tokens, max {max_new}, "
'''),
('''        _text_write(f"\\n===== end: {end}, {_time.time() - t0:.1f} s =====\\n")
''',
'''        _text_write(f"\\n===== #{rid} end: {end}, {_time.time() - t0:.1f} s =====\\n")
'''),
]
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit("anchor x%d: %s" % (s.count(old), old[:70]))
    s = s.replace(old, new)
open(P, "w").write(s)
print("patched", P)
