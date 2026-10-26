#!/usr/bin/env python3
"""tok.py - DeepSeek-V4.1 prompt <-> token ids for ds41_generate (the server's tokenizer and chat encoder).
  tok.py enc OUT.i32 "user message" [--think] [--effort N]   -> int32 ids of the chat prompt
  tok.py dec "1,2,3" | tok.py dec - (ids on stdin, comma/space separated)
"""
import json, os, sys
from pathlib import Path
import numpy as np
D = Path(os.environ.get('DS41_HOME') or Path(__file__).resolve().parent)   # where the kit is installed
X = D / 'xolotl'
sys.path[:0] = [str(X), str(X / 'tools'), str(X / 'tools/v41')]
import strata_tokenizer as ST
import dsv41_encoding as E
TD = Path(os.environ.get('DS41_MODEL_DIR') or D / 'models/gsq1017') / 'tokenizer'

def tokenizer():
    vocab = json.loads((TD / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TD / 'merges.txt').read_text(encoding='utf-8').split('\n')
    types = json.loads((TD / 'token_type.json').read_text())
    tk = ST.Tokenizer(tokens, merges, types, pre='joyai-llm')
    lit = {i: t.encode('utf-8') for t, i in tk.special_tokens.items()}
    base = tk.token_bytes
    tk.token_bytes = lambda i: lit.get(i) or base(i)
    return tk

if __name__ == '__main__':
    tk = tokenizer()
    if sys.argv[1] == 'enc':
        think = '--think' in sys.argv
        eff = None
        if '--effort' in sys.argv:
            eff = int(sys.argv[sys.argv.index('--effort') + 1])
        text = E.encode_messages([{'role': 'user', 'content': sys.argv[3]}], thinking_mode='thinking' if think else 'chat',
                                 reasoning_effort=eff)
        ids = tk.encode(text, parse_special=True)
        np.array(ids, dtype=np.int32).tofile(sys.argv[2])
        print(len(ids), 'tokens:', repr(text[:300]))
    else:
        s = sys.stdin.read() if sys.argv[2] == '-' else sys.argv[2]
        ids = [int(v) for v in s.replace(',', ' ').split()]
        print(tk.decode(ids))
