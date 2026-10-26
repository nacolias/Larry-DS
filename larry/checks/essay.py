#!/usr/bin/env python3
"""essay.py [runs] [max_tokens] - one long-essay request against the ds41 server, N times (no seed); prints the
server's decode rate per run and the mean.  The server: ds41api.py."""
import sys
import ds41api
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 800
rates = []
for i in range(N):
    body = {"model": "larry", "max_tokens": T, "reasoning_effort": "none", "messages": [{"role": "user", "content":
            "Write a detailed, long technical essay (at least 1500 words) about how virtual memory works in modern operating systems."}]}
    tm = ds41api.chat(body).get("timings", {})
    rates.append(tm.get("predicted_per_second", 0.0))
    print(f"essay run {i + 1}: {tm.get('predicted_n')} tok @ {rates[-1]:.1f} tok/s", flush=True)
print(f"essay mean {sum(rates) / len(rates):.1f} tok/s over {N} runs")
