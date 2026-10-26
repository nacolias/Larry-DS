# prov2.py - Larry's GGUF against both candidate sources, tensor by tensor
import json, struct, urllib.request
import numpy as np
GG = "/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf"
P = "/opt/ai/dsv41/logs/provenance"
REPOS = {"dealignai UNCENSORED": ("dea", "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8", "d61c59ea5e514e25d305b5850e8a432f7a9969f2"),
         "official 1017": ("off", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170")}
gi = json.load(open(P + "/gguf-infos.json"))
base, infos = gi["base"], gi["infos"]
gf = open(GG, "rb")
def gread(name, a, n):
    gf.seek(base + infos[name][2] + a)
    return gf.read(n)

def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
    return urllib.request.urlopen(req, timeout=300).read()
def hf(k, repo, sha, tensor, a=0, n=None):
    wm = json.load(open(f"{P}/idx-{k}.json"))["weight_map"]
    shard = wm[tensor]
    h, start = json.load(open(f"{P}/hdr-{k}.json"))[shard]
    o0, o1 = h[tensor]["data_offsets"]
    if n is None: n = o1 - o0 - a
    url = f"https://huggingface.co/{repo}/resolve/{sha}/{shard}"
    return get(url, start + o0 + a, start + o0 + a + n - 1)

bf16 = lambda b: (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)
lut = np.zeros(256, np.float32)
for v in range(256):
    s, e, m = v >> 7, (v >> 3) & 15, v & 7
    x = (m / 8) * 2.0 ** -6 if e == 0 else (np.nan if (e == 15 and m == 7) else (1 + m / 8) * 2.0 ** (e - 7))
    lut[v] = -x if s else x
def q8_0(b, rows, cols):
    a = np.frombuffer(b, np.uint8).reshape(rows, cols // 32, 34)
    d = a[:, :, :2].copy().view(np.float16).astype(np.float32)
    q = a[:, :, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(rows, cols)

# 1. the layer-10 router (GGUF F32 <- BF16 source: exact if it is the source)
g_gate = np.frombuffer(gread("blk.10.ffn_gate_inp.weight", 0, 384 * 5120 * 4), np.float32).reshape(384, 5120)
g_norm = np.frombuffer(gread("blk.10.attn_norm.weight", 0, 5120 * 4), np.float32)
rows = list(range(0, 32)) + list(range(50000, 50032)) + list(range(100000, 100032))
g_emb = np.stack([np.frombuffer(gread("token_embd.weight", r * 5120 * 2, 5120 * 2), np.float16).astype(np.float32) for r in rows])
g_wob = q8_0(gread("blk.10.attn_output_b.weight", 0, 5120 * 256 * 34), 5120, 8192)
print("Larry's GGUF: router", g_gate.shape, "| norm", g_norm.shape, "| embed rows", g_emb.shape, "| wo_b", g_wob.shape)
res = {}
for label, (k, repo, sha) in REPOS.items():
    s_gate = bf16(hf(k, repo, sha, "layers.10.ffn.gate.weight")).reshape(384, 5120)
    s_norm = bf16(hf(k, repo, sha, "layers.10.attn_norm.weight"))
    s_emb = np.stack([bf16(hf(k, repo, sha, "embed.weight", r * 5120 * 2, 5120 * 2)) for r in (0, 50000, 100000) for _ in [0]])
    # (three contiguous 32-row blocks)
    s_emb = np.concatenate([bf16(hf(k, repo, sha, "embed.weight", r0 * 5120 * 2, 32 * 5120 * 2)).reshape(32, 5120) for r0 in (0, 50000, 100000)])
    w8 = np.frombuffer(hf(k, repo, sha, "layers.10.attn.wo_b.weight"), np.uint8).reshape(5120, 8192)
    sc = np.frombuffer(hf(k, repo, sha, "layers.10.attn.wo_b.scale"), np.uint8).reshape(160, 256).astype(np.int32)
    scale = np.ldexp(np.float32(1), sc - 127).astype(np.float32)
    s_wob = lut[w8] * np.repeat(np.repeat(scale, 32, 0), 32, 1)
    e_emb16 = s_emb.astype(np.float16).astype(np.float32)
    r = {
        "router identical": float(np.mean(g_gate == s_gate)),
        "norm identical": float(np.mean(g_norm == s_norm)),
        "embedding identical (after f16)": float(np.mean(g_emb == e_emb16)),
        "wo_b relative error": float(np.linalg.norm(g_wob - s_wob) / np.linalg.norm(s_wob)),
    }
    res[label] = (r, s_wob)
    print(label, json.dumps(r))
# 2. a rank-1 edit hidden in the official comparison? top singular values of (GGUF - source), power iteration
for label, (r, s_wob) in res.items():
    D = (g_wob - s_wob).astype(np.float64)
    rng = np.random.default_rng(0)
    sv = []
    for kk in range(3):
        v = rng.standard_normal(D.shape[1])
        for prev in sv:
            v -= (prev[2] @ v) * prev[2]
        for _ in range(30):
            u = D @ v; u /= np.linalg.norm(u)
            v = D.T @ u
            for prev in sv:
                v -= (prev[2] @ v) * prev[2]
            s = np.linalg.norm(v); v /= s
        sv.append((s, u, v))
    print(label, "residual top singular values:", [round(float(x[0]), 4) for x in sv], "| frobenius", round(float(np.linalg.norm(D)), 4))
