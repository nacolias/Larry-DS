#!/usr/bin/env python3
"""jointcheck.py - batched decode (concurrent sessions, stage 2): two greedy answers alone (twice: today's run-to-run
spread) and then both at once (their DSpark rounds in shared TP passes); per answer the characters it shares with its
first alone run, and the decode tok/s.  Different pass widths round differently, so a long greedy answer may part
from its alone run somewhere - the second alone run shows how far that already goes.  Server: DS41_URL (ds41api.py)."""
import json, threading, time, urllib.request
import ds41api

PROMPTS = ["Write a detailed technical essay (at least 600 words) on how virtual memory works in modern operating systems.",
           "Explain step by step how to implement a hash map with open addressing in C, with complete code."]
N = 400


def run(i, out):
    body = {"model": "larry", "max_tokens": N, "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": PROMPTS[i]}]}
    h = {"Content-Type": "application/json"}
    if ds41api.key():
        h["Authorization"] = "Bearer " + ds41api.key()
    t0 = time.time()
    first, last, gen, text = None, None, 0, []
    with urllib.request.urlopen(urllib.request.Request(ds41api.URL, data=json.dumps(body).encode(), headers=h), timeout=900) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[5:])
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    last = time.time()
                    first = first or last
                    text.append(piece)
            if d.get("usage"):
                gen = d["usage"].get("completion_tokens") or gen
    out[i] = ("".join(text), gen, (gen - 1) / max(1e-9, last - first) if first else 0.0, t0, last or t0)


def common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


alone1, alone2, both = {}, {}, {}
for i in range(2):
    run(i, alone1)
for i in range(2):
    run(i, alone2)
th = [threading.Thread(target=run, args=(i, both)) for i in range(2)]
for x in th: x.start()
for x in th: x.join()
for i in range(2):
    a, b, c = alone1[i], alone2[i], both[i]
    print("answer %d: alone %d tok %.1f tok/s | again alone: %d tok %.1f tok/s, %d/%d chars as the first | at once: %d tok %.1f tok/s, %d/%d chars as the first"
          % (i, a[1], a[2], b[1], b[2], common(a[0], b[0]), len(a[0]), c[1], c[2], common(a[0], c[0]), len(a[0])))
span = max(both[i][4] for i in range(2)) - min(both[i][3] for i in range(2))
print("at once in total: %d tokens in %.1f s = %.1f tok/s" % (both[0][1] + both[1][1], span, (both[0][1] + both[1][1]) / max(span, 1e-9)))
