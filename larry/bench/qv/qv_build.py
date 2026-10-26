#!/usr/bin/env python3
"""qv_build.py TEXTS.json OUTDIR - the quant-vs-original evaluation set: 18 sequences of exactly S=2048 token ids, the
server's own tokenizer (strata_tokenizer) and chat encoder (dsv41_encoding), as tok.py uses them.
  en 4 / zh 3 / math 3: BOS + the document;  agent 4: SWE-agent trajectories rendered as a DeepSeek chat (no system
  message, thinking off);  code 4: this week's larry C++ (the Xe tier's GPU routing, the XMX indexer - unseen by any
  model), DeepSeek's own inference/model.py, CPython's argparse.py.
Writes OUTDIR/tokens.npy [N, S] int32, OUTDIR/sNN.i32 (one engine --ids-file each), OUTDIR/meta.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["/opt/ai/dsv41"]
import tok as T   # /opt/ai/dsv41/tok.py: the server's tokenizer + encoder (imports strata_tokenizer, dsv41_encoding)

S, BOS = 2048, 0


def window(path, anchor, chars=14000):
    t = Path(path).read_text(encoding="utf-8")
    i = max(0, t.find(anchor)) if anchor else 0
    return t[i:i + chars]


def main():
    texts, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tk = T.tokenizer()
    seqs, doms, names = [], [], []

    def add(dom, name, ids):
        if ids[0] != BOS:
            ids = [BOS] + ids
        assert len(ids) >= S, (dom, name, len(ids))
        seqs.append(ids[:S]); doms.append(dom); names.append(name)

    for dom in ("en", "zh", "math"):
        for i, t in enumerate(texts[dom]):
            add(dom, f"{dom}{i}:{t[:40]!r}", tk.encode(t))
    for a in texts["agent"]:
        text = T.E.encode_messages(a["messages"], thinking_mode="chat")
        add("agent", a["id"], tk.encode(text, parse_special=True))
    X = "/opt/ai/dsv41/xolotl"
    code = [("xe_tier_groute", f"{X}/tools/v41/engine/xe/ds41_xe_tier.cpp", "larry, STRATA_XE_GROUTE (default 0)"),
            ("li_xmx", f"{X}/third_party/llama.cpp/ggml/src/ggml-sycl/lightning-indexer.cpp", "static bool li_xmx_ok"),
            ("ds_model_py", "/opt/ai/dsv41/qv/ds1017/inference/model.py", "class Attention(nn.Module)"),
            ("argparse_py", "/usr/lib/python3.14/argparse.py", "class ArgumentParser(")]
    for name, path, anchor in code:
        add("code", name, tk.encode(window(path, anchor)))
    tokens = np.array(seqs, dtype=np.int32)
    np.save(out / "tokens.npy", tokens)
    for i, s in enumerate(tokens):
        s.tofile(out / f"s{i:02d}.i32")
    json.dump({"n": len(seqs), "seq": S, "domains": doms, "names": names, "is_eval": [True] * len(seqs)},
              open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
    for i, (d, n) in enumerate(zip(doms, names)):
        print(f"s{i:02d} {d:5s} {n[:70]}  | {tk.decode(list(tokens[i][1:24]))!r}")


if __name__ == "__main__":
    main()
