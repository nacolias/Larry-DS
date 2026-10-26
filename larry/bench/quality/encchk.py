#!/usr/bin/env python3
"""encchk.py - does the ds41 server render an AIME prompt at reasoning effort 100 the way DeepSeek's eval does?
DeepInfer's eval renders encode_messages(messages, thinking_mode="thinking", reasoning_effort=100) with the
checkpoint's encoding/encoding.py; the server renders with Xolotl's dsv41_encoding.  Prints both texts' token counts
and whether they are identical (run with /opt/ai/dsv41/venv/bin/python -I)."""
import importlib.util, sys
sys.path.insert(0, '/opt/ai/dsv41')
import tok
SNAP = '/opt/ai/dsv41/bench/speed/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170'
spec = importlib.util.spec_from_file_location('ckpt_encoding', SNAP + '/encoding/encoding.py')
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)
q = 'Find the number of ordered pairs.\nPlease reason step by step, and put your final answer within \\boxed{}.'
msgs = [{'role': 'user', 'content': q}]
tk = tok.tokenizer()
for eff in (100, 'max', 'high', None):
    try:
        a = O.encode_messages(msgs, thinking_mode='thinking', reasoning_effort=eff)
    except Exception as e:
        a = 'ERR %r' % e
    try:
        b = tok.E.encode_messages(msgs, thinking_mode='thinking', reasoning_effort=eff)
    except Exception as e:
        b = 'ERR %r' % e
    print('effort %r: official %d tokens, xolotl %d tokens, identical %s' % (
        eff, len(tk.encode(a, parse_special=True)), len(tk.encode(b, parse_special=True)), a == b))
    if eff == 100:
        print(repr(a[:600]))
