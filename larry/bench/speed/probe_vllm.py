#!/usr/bin/env python3
"""probe_vllm.py PORT - one MT-Bench-style request through vLLM 0.30's (patched) chat request function, with every
network read and parsed message timestamped: how vLLM's client sees the ds41 server's stream."""
import asyncio, json, sys, time
import aiohttp
import vllm.benchmarks.lib.endpoint_request_func as m

orig = m.StreamedResponseHandler.add_chunk
T0 = [0.0]


def add_chunk(self, b):
    msgs = orig(self, b)
    def brief(s):
        try:
            d = json.loads(s.removeprefix('data: '))
        except ValueError:
            return s[:40]
        c = (d.get('choices') or [{}])[0]
        return {'delta': c.get('delta'), 'fin': c.get('finish_reason'), 'usage': d.get('usage')}
    print('%8.1f ms read %4d B -> %s | buffer %r' % ((time.perf_counter() - T0[0]) * 1e3, len(b),
          [brief(s) for s in msgs], self.buffer[:40]))
    return msgs


m.StreamedResponseHandler.add_chunk = add_chunk


async def main():
    inp = m.RequestFuncInput(prompt='Compose an engaging travel blog post about a recent trip to %s, highlighting '
                             'cultural experiences and must-see attractions.' % sys.argv[2], api_url='http://172.23.0.1:%s/v1/chat/'
                             'completions' % sys.argv[1], prompt_len=30, output_len=24, model='deepseek-v4.1-flash-1017',
                             extra_body={'reasoning_effort': 'none', 'temperature': 1.0, 'top_p': 0.95})
    async with aiohttp.ClientSession() as s:
        T0[0] = time.perf_counter()
        out = await m.async_request_openai_chat_completions(inp, s)
    print('ttft %.1f ms, output_tokens %s, itl n %d, text %r, error %r' % (out.ttft * 1e3, out.output_tokens,
          len(out.itl), out.generated_text[:100], out.error))

asyncio.run(main())
