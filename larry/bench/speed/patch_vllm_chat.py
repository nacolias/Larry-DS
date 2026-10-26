#!/usr/bin/env python3
"""patch_vllm_chat.py ORIG OUT - vLLM 0.30's benchmark client (vllm/benchmarks/lib/endpoint_request_func.py, the copy in
/opt/venv/lib/python3.12/site-packages that `vllm bench serve` imports), the chat backend only:

1. a streamed chunk counts as a token only when its delta carries text.  The ds41 server sends the OpenAI role chunk
   ({"role": "assistant", "content": ""}) as soon as a request arrives, before the prompt is read (vLLM's own server
   sends it with the first token), and vLLM's client timed TTFT to the first chunk of any kind - a few ms here.  Its
   finish chunk (an empty delta + usage) also counted as one more token gap and kept the `elif` that reads the usage
   from running, so the output length came from re-tokenising the text.
2. the network reads go to the SSE parser unstripped.  The client stripped each read before buffering it, which
   removed the blank line that ends an event: a ": keep-alive" comment (the ds41 server sends them while it reads a
   prompt) that arrived alone stayed in the buffer, the next events were appended to it and dropped with it as one
   comment - the first words of every answer.  vLLM's own server sends no comments.
"""
import sys

src = open(sys.argv[1]).read()
start = src.index('async def async_request_openai_chat_completions(')
end = src.index('\nasync def ', start + 10)
body = src[start:end]
for old, new in (
        ('                            if choices := data.get("choices"):\n',
         '                            if (choices := data.get("choices")) and (\n'
         '                                choices[0]["delta"].get("content")\n'
         '                                or choices[0]["delta"].get("reasoning_content")\n'
         '                            ):\n'),
        ('                    chunk_bytes = chunk_bytes.strip()\n'
         '                    if not chunk_bytes:\n'
         '                        continue\n',
         '                    if not chunk_bytes.strip():\n'
         '                        continue\n')):
    assert body.count(old) == 1, (old, body.count(old))
    body = body.replace(old, new)
open(sys.argv[2], 'w').write(src[:start] + body + src[end:])
print('patched', sys.argv[2])
