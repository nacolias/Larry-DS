#!/usr/bin/env python3
"""svcbench.py [max_tokens] - a few chat requests against the ds41 server (DSpark sampling, the served settings);
prints the server's own timings per request and the mean decode rate.  The server: ds41api.py."""
import sys, time
import ds41api
MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PROMPTS = [
    ("code", "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, with tests."),
    ("prose", "Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle."),
    ("explain", "Explain how a B-tree differs from a binary search tree and why databases use B-trees. Be concrete."),
    ("cpp", "Implement a thread-safe bounded queue in C++17 with push, pop and try_pop, and explain the design."),
]
rates = []
for name, text in PROMPTS:
    body = {"model": "larry", "max_tokens": MAXT, "reasoning_effort": "none",
            "seed": 7, "messages": [{"role": "user", "content": text}]}
    t = time.time()
    d = ds41api.chat(body)
    tm = d.get("timings", {})
    r = tm.get("predicted_per_second", 0.0)
    rates.append(r)
    out = (d["choices"][0]["message"]["content"] or "").replace("\n", " ")
    print(f"{name:8s} {time.time() - t:6.1f} s  prompt {tm.get('prompt_n')} tok @ {tm.get('prompt_per_second', 0):.0f}/s, "
          f"decode {tm.get('predicted_n')} tok @ {r:.1f} tok/s | {out[:90]!r}")
print(f"mean decode rate {sum(rates) / len(rates):.1f} tok/s")
