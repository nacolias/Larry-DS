#!/usr/bin/env python3
"""qv_engram_small.py OUTDIR - the Engram layers' small tensors (k_weight, q_weight, wkv.weight, wkv.scale) of
DeepSeek-V4.1-Flash-1017's shards 47 / 48 by HTTP range requests (the shards are 101.5 GB each, ~all of it the
embedding tables we already have extracted), written as OUTDIR/model-0004{7,8}-of-00048.safetensors (+ .ok) with
only those tensors.  The tables themselves stay in the engine's extracted files (same bytes: extract_engram.py)."""
import json, os, struct, sys, urllib.request

URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"


def get(name, a, b):
    req = urllib.request.Request(URL + name, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        assert r.status == 206, r.status
        data = r.read()
    assert len(data) == b - a, (name, a, b, len(data))
    return data


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        n = struct.unpack("<Q", get(name, 0, 8))[0]
        hdr = json.loads(get(name, 8, 8 + n))
        hdr.pop("__metadata__", None)
        base = 8 + n
        keep = {k: v for k, v in hdr.items() if ".engram.embed." not in k}
        new_hdr, blobs, off = {}, [], 0
        for k, v in sorted(keep.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = v["data_offsets"]
            blobs.append(get(name, base + a, base + b))
            new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + (b - a)]}
            off += b - a
        h = json.dumps(new_hdr, separators=(",", ":")).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        path = os.path.join(out, name)
        with open(path + ".part", "wb") as f:
            f.write(struct.pack("<Q", len(h))); f.write(h)
            for blob in blobs:
                f.write(blob)
        os.replace(path + ".part", path)
        open(path + ".ok", "w").write("range-fetched: " + ",".join(sorted(new_hdr)) + "\n")
        print(name, {k: (v["dtype"], v["shape"]) for k, v in new_hdr.items()}, off, "bytes")


if __name__ == "__main__":
    main()
