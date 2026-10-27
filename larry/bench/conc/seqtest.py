#!/usr/bin/env python3
"""seqtest.py ENGINE_LOG - concurrency stage 1 step A (STRATA_SERVE_SEQS=2): two conversations resident on the cards.
X (a ~31K-token needle) and Y (~8K) are read once; their follow-ups then alternate X, Y, X, Y - each must continue its
own conversation slot (reused = the whole conversation, no unpark, no re-read).  Then Z (a new ~6K prompt) takes the
slot used longest ago (X's or Y's conversation parked into the pool), and that conversation's next turn comes back
from the pool.  Every answer checked; the engine log's reuse lines (seq=) and pool lines shown.  Server: DS41_URL."""
import glob, re, sys, time
import ds41api

LOG = sys.argv[1]
src = ''.join(open(f).read() for f in sorted(glob.glob('/opt/ai/dsv41/xolotl/tools/v41/engine/*.cpp')))
fails = 0


def doc(off, tokens, pw):
    chars = tokens * 3
    f = (src * 3)[off:off + chars]
    mid = len(f) // 2
    return f[:mid] + "\n// NOTE FOR REVIEWERS: the deployment passphrase is '%s'.\n" % pw + f[mid:]


def ask(tag, msgs, pw, n=24, want_reuse=0):
    global fails
    t = time.time()
    d = ds41api.chat({"model": "larry", "messages": msgs, "max_tokens": n, "temperature": 0, "reasoning_effort": "none"})
    a = d["choices"][0]["message"].get("content") or ""
    tm = d.get("timings", {})
    ok = pw is None or pw in a
    reused = tm.get("cache_n") or 0
    if want_reuse and reused < want_reuse:
        ok = False
    fails += 0 if ok else 1
    print("%-30s %6.1f s  read %6s  reused %6s  %-5s %r" % (tag, time.time() - t, tm.get("prompt_n"), reused,
                                                         "right" if ok else "WRONG", a[:50]), flush=True)
    return a


log0 = sum(1 for _ in open(LOG, errors="replace"))
conv = {}
for name, off, n, pw in (("X", 0, 31000, "violet-kestrel-4217"), ("Y", 400000, 8000, "amber-falcon-9034")):
    msgs = [{"role": "user", "content": "Conversation %s. Here is a source dump:\n\n%s\n\nWhat is the deployment passphrase? "
                                       "Answer with the passphrase only." % (name, doc(off, n, pw))}]
    a = ask("%s turn 1 (~%dK, fresh)" % (name, n // 1000), msgs, pw)
    conv[name] = (msgs + [{"role": "assistant", "content": a}], pw, n)
for rnd in (2, 3):
    for name in ("X", "Y"):
        msgs, pw, n = conv[name]
        msgs.append({"role": "user", "content": "Say the passphrase again, then the word DONE (turn %d)." % rnd})
        a = ask("%s turn %d (resident)" % (name, rnd), msgs, pw, want_reuse=int(n * 0.9))
        msgs.append({"role": "assistant", "content": a})
z = [{"role": "user", "content": "Conversation Z. Here is a source dump:\n\n%s\n\nWhat is the deployment passphrase? Answer "
                                 "with the passphrase only." % doc(800000, 6000, "cobalt-heron-5561")}]
ask("Z turn 1 (~6K, fresh)", z, "cobalt-heron-5561")
for name in ("X", "Y"):   # one of them was parked for Z: it comes back from the pool; the other is still resident
    msgs, pw, n = conv[name]
    msgs.append({"role": "user", "content": "Say the passphrase again, then the word DONE (turn 4)."})
    ask("%s turn 4 (resident or pool)" % name, msgs, pw, want_reuse=int(n * 0.9))
lines = list(open(LOG, errors="replace"))[log0:]
for ln in lines:
    if re.search(r"serve: reuse lcp|unparked the|conversation cache:|conversation slots", ln):
        print("   " + ln.rstrip()[:170])
print("RESULT", "PASS" if fails == 0 else "FAIL (%d)" % fails)
