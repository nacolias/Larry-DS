#!/usr/bin/env python3
"""needle.py [approx_tokens] [fresh] - a passphrase hidden in the middle of ~N tokens of filler; asks for it (ds41).
fresh=1: a random first line, so no earlier prompt is a prefix of this one (a cold prompt every time).  The filler
is the engine's own C++ (DS41_HOME/xolotl/tools/v41/engine/*.cpp); the server: ds41api.py."""
import glob, os, random, sys, time
import ds41api
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
FRESH = len(sys.argv) > 2 and sys.argv[2] == '1'
src = ''
for f in sorted(glob.glob(os.path.join(ds41api.D, 'xolotl/tools/v41/engine/*.cpp'))):
    src += open(f).read()
chars = N * 3                      # ~3 characters a token for C++
filler = (src * (chars // len(src) + 1))[:chars]
mid = len(filler) // 2
needle = "\n// NOTE FOR REVIEWERS: the deployment passphrase is 'violet-kestrel-4217'.\n"
text = filler[:mid] + needle + filler[mid:]
body = {"model": "larry", "max_tokens": 200, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": ("Session %08x. " % random.getrandbits(32) if FRESH else "") +
                      "Here is a large source dump:\n\n" + text +
                      "\n\nSomewhere in it is a deployment passphrase. What is it? Answer with the passphrase only."}]}
t = time.time()
d = ds41api.chat(body, timeout=3600)
print(f"{time.time() - t:.1f} s:", repr(d["choices"][0]["message"]["content"]))
print(d["usage"], {k: d["timings"][k] for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_per_second")})
print("FOUND" if "violet-kestrel-4217" in (d["choices"][0]["message"]["content"] or "") else "MISSED")
