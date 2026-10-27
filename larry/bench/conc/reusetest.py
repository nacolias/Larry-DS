#!/usr/bin/env python3
"""reusetest.py ENGINE_LOG - conversation reuse through the park pool (concurrency stage 0, patch-parkpool.py).
Four conversations A-D (~3.0-3.9K tokens, each with its own passphrase) asked in turn, then a follow-up on each in turn:
each should come back from the pool (unparked whole: a few new tokens read, not the whole prompt).  Then E is asked,
an unrelated side request runs, and E is asked again with its last words changed (a re-rendered tail within 16 tokens
of the park's end: the partial resume) and with a change ~200 tokens back (read fresh now).  Every answer is checked,
and the engine log's park / unpark lines are counted.  Server: DS41_URL (ds41api.py)."""
import glob, re, sys, time
import ds41api

LOG = sys.argv[1]
src = ''.join(open(f).read() for f in sorted(glob.glob('/opt/ai/dsv41/xolotl/tools/v41/engine/*.cpp')))
PW = ["violet-kestrel-4217", "amber-falcon-9034", "cobalt-heron-5561", "jade-osprey-2289", "scarlet-wren-7710"]
fails = 0


def doc(i, tokens):
    chars = tokens * 3                 # ~3 characters a token for C++
    off = (i * 100003) % (len(src) - chars - 1)
    f = src[off:off + chars]
    mid = len(f) // 2
    return f[:mid] + "\n// NOTE FOR REVIEWERS: the deployment passphrase is '%s'.\n" % PW[i] + f[mid:]


def ask(tag, msgs, pw, n=24):
    global fails
    t = time.time()
    d = ds41api.chat({"model": "larry", "messages": msgs, "max_tokens": n, "temperature": 0, "reasoning_effort": "none"})
    a = d["choices"][0]["message"].get("content") or ""
    tm = d.get("timings", {})
    ok = pw is None or pw in a
    fails += 0 if ok else 1
    print("%-34s %5.1f s  read %6s  reused %6s  %-5s %r" % (tag, time.time() - t, tm.get("prompt_n"), tm.get("cache_n"),
                                                         "right" if ok else "WRONG", a[:60]))
    return a, tm


def q(i, text, tail="Answer with the passphrase only."):
    return {"role": "user", "content": "Conversation %d. Here is a source dump:\n\n%s\n\nWhat is the deployment "
                                       "passphrase? %s" % (i, text, tail)}


log0 = sum(1 for _ in open(LOG, errors="replace"))
convs = []
for i, n in enumerate([3000, 3300, 3600, 3900]):
    msgs = [q(i, doc(i, n))]
    a, _ = ask("turn 1 %s (%d tokens)" % ("ABCD"[i], n), msgs, PW[i])
    convs.append(msgs + [{"role": "assistant", "content": a}])
for i, msgs in enumerate(convs):
    msgs.append({"role": "user", "content": "Say the passphrase again, then the word DONE."})
    _, tm = ask("turn 2 %s (from the pool)" % "ABCD"[i], msgs, PW[i])
    if (tm.get("cache_n") or 0) < 2048:
        print("   ^ NOT REUSED")
        fails += 1
e = doc(4, 3000)
ask("turn 1 E", [q(4, e)], PW[4])
ask("side request (unrelated)", [{"role": "user", "content": "Name three prime numbers. " + src[5000:5600]}], None)
_, tm = ask("E, last words re-rendered", [q(4, e, "Reply with the passphrase only.")], PW[4])
if (tm.get("cache_n") or 0) < 2048:
    print("   ^ NOT REUSED (the partial resume)")
    fails += 1
ask("E, changed ~200 tokens back", [q(4, e + "\n// (end of the dump; it has about three thousand tokens)\n" * 12)], PW[4])
lines = list(open(LOG, errors="replace"))[log0:]
cnt = {k: sum(1 for ln in lines if k in ln) for k in ("conversation cache: parked", "conversation cache: restored",
                                                     "partial: re-feeds", "superseded", "for the ", "skipped")}
print("engine log:", cnt)
for ln in lines:
    if re.search(r"unparked the|conversation cache:", ln):
        print("   " + ln.rstrip()[:150])
print("RESULT", "PASS" if fails == 0 and cnt["conversation cache: restored"] >= 5 and cnt["partial: re-feeds"] >= 1
      else "FAIL (%d wrong / unreused)" % fails)
