#!/usr/bin/env python3
"""ctxsvc.py [approx_tokens] [runs] - the essay request behind a ~N-token source-dump prefix, against the larry ds41
service: decode speed at a long context.  The first run prefills the prefix; later runs reuse it (prefix cache), as
agentic turns do.  Prints the server's prompt / decode timings per run."""
import glob, json, sys, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 95000
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
KEY = [l.split('=', 1)[1].strip() for l in open('/opt/docker/vllm/.env') if l.startswith('VLLM_API_KEY')][0]
src = ''
for f in sorted(glob.glob('/opt/ai/dsv41/xolotl/tools/v41/engine/*.cpp')):
    src += open(f).read()
chars = N * 3                      # ~3 characters a token for C++
text = (src * (chars // len(src) + 1))[:chars]
for i in range(R):
    body = {"model": "larry", "max_tokens": 400, "reasoning_effort": "none", "messages": [{"role": "user", "content":
            "Here is a large source dump:\n\n" + text + "\n\nIgnore it. Write a detailed, long technical essay (at least "
            "1500 words) about how virtual memory works in modern operating systems."}]}
    req = urllib.request.Request('http://172.23.0.1:18080/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    tm = {k: (v if v is not None else 0) for k, v in d.get("timings", {}).items()}
    print(f"run {i + 1}: prompt {d['usage'].get('prompt_tokens')} tok ({tm.get('prompt_n')} new, "
          f"{tm.get('prompt_per_second', 0):.1f} tok/s), decode {tm.get('predicted_n')} tok @ "
          f"{tm.get('predicted_per_second', 0):.1f} tok/s", flush=True)
