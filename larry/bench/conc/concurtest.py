#!/usr/bin/env python3
"""concurtest.py ENGINE_LOG [ROUNDS] - concurrent sessions (stage 1, "parallel": N through serve/server.py's batch client).
Round 1: a long answer (an essay, 600 tokens) starts alone; 3 s later a short question (a ~4K-token needle) arrives - it
must not wait for the essay's end (its first token comes while the essay still decodes).  Round 2: two needle turns
(~8K and ~12K, different conversations) and the essay's follow-up at once.  Then the follow-ups of both needles at once
(each must come back from its own conversation slot: reused).  Every answer checked; per request: wall time, first
token, tokens, tok/s; the engine log's slot lines.  Server: DS41_URL (ds41api.py)."""
import glob, json, re, sys, threading, time, urllib.request
import ds41api

LOG = sys.argv[1]
src = ''.join(open(f).read() for f in sorted(glob.glob('/opt/ai/dsv41/xolotl/tools/v41/engine/*.cpp')))
fails = 0
lock = threading.Lock()


def doc(off, tokens, pw):
    f = (src * 3)[off:off + tokens * 3]
    mid = len(f) // 2
    return f[:mid] + "\n// NOTE FOR REVIEWERS: the deployment passphrase is '%s'.\n" % pw + f[mid:]


def stream(tag, msgs, n, check=None, out=None):
    """one streamed request: first-token time, tokens, the text"""
    global fails
    body = {"model": "larry", "messages": msgs, "max_tokens": n, "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if ds41api.key():
        h["Authorization"] = "Bearer " + ds41api.key()
    req = urllib.request.Request(ds41api.URL, data=json.dumps(body).encode(), headers=h)
    t0 = time.time()
    first, text, usage, ntok = None, "", {}, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[5:])
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    if first is None:
                        first = time.time() - t0
                    text += piece
                    ntok += 1
            if d.get("usage"):
                usage = d["usage"]
    wall = time.time() - t0
    ok = check is None or check(text)
    gen = usage.get("completion_tokens") or ntok
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    with lock:
        fails += 0 if ok else 1
        print("%-26s start %5.1f  first %5.1f s  done %6.1f s  %4d tok  %5.1f tok/s  cached %6d  %-5s %r" % (
            tag, t0 - T0, first or -1, wall, gen, gen / max(wall - (first or 0), 1e-9), cached, "right" if ok else "WRONG",
            text[:50].replace("\n", " ")), flush=True)
    if out is not None:
        out.append(text)
    return text


def run(jobs):
    th = []
    for delay, args in jobs:
        def go(a=args, dl=delay):
            time.sleep(dl)
            stream(*a)
        t = threading.Thread(target=go)
        t.start()
        th.append(t)
    for t in th:
        t.join()


log0 = sum(1 for _ in open(LOG, errors="replace"))
T0 = time.time()
essay = [{"role": "user", "content": "Write a detailed technical essay (about 500 words) on how virtual memory works."}]
n1 = [{"role": "user", "content": "Here is a source dump:\n\n%s\n\nWhat is the deployment passphrase? Answer with the "
                                  "passphrase only." % doc(0, 4000, "violet-kestrel-4217")}]
ess_out = []
print("-- round 1: an essay alone, a short needle 3 s later")
run([(0, ("essay (600)", essay, 600, lambda t: len(t) > 1000, ess_out)),
     (3, ("needle 4K (arrives 3 s)", n1, 24, lambda t: "violet-kestrel-4217" in t))])
print("-- round 2: two needles and the essay's follow-up at once")
n2 = [{"role": "user", "content": "Here is another source dump:\n\n%s\n\nWhat is the deployment passphrase? Answer with "
                                  "the passphrase only." % doc(300000, 8000, "amber-falcon-9034")}]
n3 = [{"role": "user", "content": "One more source dump:\n\n%s\n\nWhat is the deployment passphrase? Answer with the "
                                  "passphrase only." % doc(600000, 12000, "cobalt-heron-5561")}]
ess2 = essay + [{"role": "assistant", "content": ess_out[0] if ess_out else ""},
                {"role": "user", "content": "Now summarise that essay in three sentences."}]
o2, o3 = [], []
run([(0, ("needle 8K", n2, 24, lambda t: "amber-falcon-9034" in t, o2)),
     (0, ("needle 12K", n3, 24, lambda t: "cobalt-heron-5561" in t, o3)),
     (0, ("essay follow-up (200)", ess2, 200, lambda t: len(t) > 200))])
print("-- round 3: both needles' follow-ups at once (each from its own conversation)")
f2 = n2 + [{"role": "assistant", "content": o2[0] if o2 else ""}, {"role": "user", "content": "Say it again, then DONE."}]
f3 = n3 + [{"role": "assistant", "content": o3[0] if o3 else ""}, {"role": "user", "content": "Say it again, then DONE."}]
run([(0, ("needle 8K follow-up", f2, 24, lambda t: "amber-falcon-9034" in t)),
     (0, ("needle 12K follow-up", f3, 24, lambda t: "cobalt-heron-5561" in t))])
lines = list(open(LOG, errors="replace"))[log0:]
for ln in lines:
    if re.search(r"serve: slot |serve: reuse lcp|conversation cache:|rewound", ln):
        print("   " + ln.rstrip()[:170])
print("RESULT", "PASS" if fails == 0 else "FAIL (%d)" % fails)
