#!/usr/bin/env python3
"""twoessays.py [N] - the decode rate of two long answers at once (each N tokens, default 500, greedy, different topics),
against one alone: per stream and in total.  Server: DS41_URL (ds41api.py)."""
import json, sys, threading, time, urllib.request
import ds41api

TOPICS = ["how virtual memory works in modern operating systems", "how TCP congestion control works",
          "how a garbage collector with generations works"]


N = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def run(topic, out, n=N):
    body = {"model": "larry", "max_tokens": n, "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Write a detailed technical essay (at least 600 words) on " + topic + "."}]}
    h = {"Content-Type": "application/json"}
    if ds41api.key():
        h["Authorization"] = "Bearer " + ds41api.key()
    t0 = time.time()
    first, last, gen = None, None, 0
    with urllib.request.urlopen(urllib.request.Request(ds41api.URL, data=json.dumps(body).encode(), headers=h), timeout=900) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[5:])
            for c in d.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    now = time.time()
                    first = first or now
                    last = now
            if d.get("usage"):
                gen = d["usage"].get("completion_tokens") or gen
    out.append((topic, gen, first - t0, last - first, t0, last))


solo = []
run(TOPICS[2], solo)
t, g, ft, dt, _, _ = solo[0]
print("alone:         %4d tok, first %.1f s, %.1f tok/s" % (g, ft, (g - 1) / dt))
res = []
th = [threading.Thread(target=run, args=(TOPICS[i], res)) for i in range(2)]
for x in th: x.start()
for x in th: x.join()
for t, g, ft, dt, a, b in res:
    print("two at once:   %4d tok, first %.1f s, %.1f tok/s  (%s)" % (g, ft, (g - 1) / dt, t[:30]))
both0 = max(r[4] + r[2] for r in res)
both1 = min(r[5] for r in res)
tok = sum(r[1] for r in res)
span = max(r[5] for r in res) - min(r[4] for r in res)
print("in total:      %d tokens in %.1f s = %.1f tok/s (both decoding %.1f s of it)" % (tok, span, tok / span, max(0.0, both1 - both0)))
