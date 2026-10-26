# prov1.py - where the comparable tensors are: the GGUF's tensor infos, and each HF repo's safetensors headers
import json, struct, sys, urllib.request
GG = "/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf"
P = "/opt/ai/dsv41/logs/provenance"
REPOS = {"dea": ("dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8", "d61c59ea5e514e25d305b5850e8a432f7a9969f2"),
         "off": ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170")}
WANT = ["embed.weight", "layers.10.attn.wo_b.weight", "layers.10.attn.wo_b.scale", "layers.10.ffn.gate.weight",
        "layers.10.attn_norm.weight", "layers.10.attn.wq_a.weight"]

f = open(GG, "rb")
magic, ver, nt, nkv = struct.unpack("<IIQQ", f.read(24))
def rs():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skip(t):
    if t == 8: rs(); return
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n): skip(et)
        return
    f.read(SZ[t])
align = 32
for _ in range(nkv):
    k = rs(); t, = struct.unpack("<I", f.read(4))
    if k == "general.alignment":
        align, = struct.unpack("<I", f.read(4)); continue
    skip(t)
infos = {}
for _ in range(nt):
    name = rs(); nd, = struct.unpack("<I", f.read(4))
    dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd)); ty, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8))
    infos[name] = (dims, ty, off)
base = (f.tell() + align - 1) // align * align
json.dump({"base": base, "infos": infos}, open(P + "/gguf-infos.json", "w"))
print("gguf data base", base, "align", align)
for n in sorted(infos):
    if n.startswith("blk.10.") and "exps" not in n or n.startswith("token_embd"):
        print("  gguf", n, infos[n][0], "type", infos[n][1])

def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
    return urllib.request.urlopen(req, timeout=120).read()
for k, (repo, sha) in REPOS.items():
    wm = json.load(open(f"{P}/idx-{k}.json"))["weight_map"]
    hdrs = {}
    for w in WANT:
        shard = wm[w]
        url = f"https://huggingface.co/{repo}/resolve/{sha}/{shard}"
        if shard not in hdrs:
            n, = struct.unpack("<Q", get(url, 0, 7))
            hdrs[shard] = (json.loads(get(url, 8, 8 + n - 1)), 8 + n)
        h, start = hdrs[shard]
        e = h[w]
        print(k, w, shard, e["dtype"], e["shape"], "bytes", e["data_offsets"][1] - e["data_offsets"][0])
    json.dump({s: v for s, v in hdrs.items()}, open(f"{P}/hdr-{k}.json", "w"))
