#!/usr/bin/env python3
"""qv_prep.py CORPUS_DIR - the teacher's inputs in $V41_WORK from qv_build.py's corpus: tokens.npy, meta.json and the
Engram n-gram hash ids (DeepSeek's NgramHashState over the same ids; v41_calib.py prep-ids for N sequences), and
MAIN's weight index with the Engram tables pointed at the engine's extracted files (engram-L1 / engram-L14)."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, "/opt/ai/dsv41/xolotl/tools/v41")
import torch  # noqa: E402
import v41_calib as C  # noqa: E402


def main():
    src = sys.argv[1]
    os.makedirs(C.WORK, exist_ok=True)
    assert not os.path.exists(C.wpath("h", "L00.bin")), f"{C.WORK} already has a chain"
    idx = os.path.join(C.MAIN, "model.safetensors.index.json")
    if not os.path.exists(idx + ".orig"):
        shutil.copy(idx, idx + ".orig")
    d = json.load(open(idx + ".orig"))
    for L, f in ((1, "engram-L1.safetensors"), (14, "engram-L14.safetensors")):
        for t in ("weight", "scale"):
            k = f"layers.{L}.engram.embed.{t}"
            assert k in d["weight_map"], k
            d["weight_map"][k] = f
    json.dump(d, open(idx, "w"))
    tokens = np.load(os.path.join(src, "tokens.npy"))
    meta = json.load(open(os.path.join(src, "meta.json")))
    N, S = tokens.shape
    np.save(C.wpath("tokens.npy"), tokens)
    json.dump({"n": N, "seq": S, "domains": meta["domains"], "names": meta["names"], "is_eval": [True] * N,
               "corpus": src}, open(C.wpath("meta.json"), "w"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MAIN)
    args = C.model_args(N, S)
    layout = C.M.EngramLayout.from_args(args)
    hs = C.M.NgramHashState(args, layout, tok)
    with torch.inference_mode():
        hashes = hs(torch.from_numpy(tokens).long(), 0)
    np.save(C.wpath("hashes.npy"), hashes.numpy())
    C.log(f"qv_prep: {tokens.shape} tokens, hashes {tuple(hashes.shape)} max id {int(hashes.max())} "
          f"(tables {args.engram_num_embeddings})")


if __name__ == "__main__":
    main()
