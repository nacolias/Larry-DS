#!/usr/bin/env python3
"""wbessay.py TOPIC_INDEX [TOKENS] - one greedy essay (twoessays.py's topics) through DS41_URL (ds41api.py): its tokens and
decode tok/s.  For wb.sh: an answer of prose before the width bench (its conversation then serves as the other one)."""
import json, sys, time, urllib.request
import ds41api

TOPICS = ["how virtual memory works in modern operating systems", "how TCP congestion control works",
          "how a garbage collector with generations works"]
topic = TOPICS[int(sys.argv[1]) % len(TOPICS)]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
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
                last = time.time()
                first = first or last
        if d.get("usage"):
            gen = d["usage"].get("completion_tokens") or gen
print("%d tokens, %.1f tok/s (%s)" % (gen, (gen - 1) / max(1e-9, last - first) if first else 0.0, topic[:40]))
