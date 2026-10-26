#!/usr/bin/env python3
"""effchk.py - does a request's reasoning_effort reach the engine's prompt?  Two 1-token requests that differ only in
the effort ("max" = 100, then 50); the engine log then shows the second prompt reusing only the tokens before the
effort digits (identical prompts would take the regen path)."""
import http.client, json
key = [l.split('=', 1)[1].strip() for l in open('/opt/docker/vllm/.env') if l.startswith('VLLM_API_KEY')][0]
for eff in ('max', 50):
    c = http.client.HTTPConnection('172.23.0.1', 18080, timeout=120)
    c.request('POST', '/v1/chat/completions', json.dumps({
        'model': 'x', 'messages': [{'role': 'user', 'content': 'What is 2+2? Please reason step by step.'}],
        'max_tokens': 1, 'reasoning_effort': eff, 'temperature': 1.0, 'top_p': 1.0}),
        {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    r = json.loads(c.getresponse().read())
    print(eff, r.get('usage'), (r.get('timings') or {}).get('cache_n'))
