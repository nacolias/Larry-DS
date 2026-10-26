# prov3.py - every layer: Larry's GGUF against the official DeepSeek-V4.1-Flash-1017 weights.  Per layer the router
# (exact: F32 here, BF16 there), the attention output projection wo_b and the shared expert's down projection w2 (Q8_0
# here, FP8 + E8M0 block scales there): the relative difference, and the top singular values of the difference
# (an abliteration's rank-1 edit would stand out as one large value over a flat quantization-noise floor).
import json, struct, urllib.request
import numpy as np
GG = "/opt/ai/dsv41/models/gsq1017/V41-GSQ-1017.gguf"
P = "/opt/ai/dsv41/logs/provenance"
REPO, SHA = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
gi = json.load(open(P + "/gguf-infos.json"))
base, infos = gi["base"], gi["infos"]
gf = open(GG, "rb")
def gread(name, n):
    gf.seek(base + infos[name][2]); return gf.read(n)
wm = json.load(open(P + "/idx-off.json"))["weight_map"]
hdrs = {}
def get(url, a, b):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
            return urllib.request.urlopen(req, timeout=300).read()
        except Exception as e:
            err = e
    raise err
def hf(tensor):
    shard = wm[tensor]
    url = f"https://huggingface.co/{REPO}/resolve/{SHA}/{shard}"
    if shard not in hdrs:
        n, = struct.unpack("<Q", get(url, 0, 7))
        hdrs[shard] = (json.loads(get(url, 8, 8 + n - 1)), 8 + n)
    h, start = hdrs[shard]
    o0, o1 = h[tensor]["data_offsets"]
    return get(url, start + o0, start + o1 - 1), h[tensor]["shape"]
bf16 = lambda b: (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)
lut = np.zeros(256, np.float32)
for v in range(256):
    s, e, m = v >> 7, (v >> 3) & 15, v & 7
    x = (m / 8) * 2.0 ** -6 if e == 0 else (np.nan if (e == 15 and m == 7) else (1 + m / 8) * 2.0 ** (e - 7))
    lut[v] = -x if s else x
def q8_0(b, rows, cols):
    a = np.frombuffer(b, np.uint8).reshape(rows, cols // 32, 34)
    d = a[:, :, :2].copy().view(np.float16).astype(np.float32)
    return (d * a[:, :, 2:].view(np.int8).astype(np.float32)).reshape(rows, cols)
def fp8(wname):
    wb, shp = hf(wname)
    sb, sshp = hf(wname.replace(".weight", ".scale"))
    w8 = np.frombuffer(wb, np.uint8).reshape(shp)
    sc = np.ldexp(np.float32(1), np.frombuffer(sb, np.uint8).reshape(sshp).astype(np.int32) - 127).astype(np.float32)
    br, bc = shp[0] // sshp[0], shp[1] // sshp[1]
    return lut[w8] * np.repeat(np.repeat(sc, br, 0), bc, 1)
def top_sv(D, k=3, it=20):
    rng = np.random.default_rng(0); sv = []
    for _ in range(k):
        v = rng.standard_normal(D.shape[1])
        for _ in range(it):
            for _, _, pv in sv: v -= (pv @ v) * pv
            u = D @ v; u /= np.linalg.norm(u)
            v = D.T @ u
            for _, _, pv in sv: v -= (pv @ v) * pv
            s = np.linalg.norm(v); v /= s
        sv.append((s, u, v))
    return [float(x[0]) for x in sv]
print("layer | router identical | wo_b rel.diff  top-3 sv of diff | shexp w2 rel.diff  top-3 sv of diff", flush=True)
worst = []
for L in range(40):
    out = [str(L)]
    g = np.frombuffer(gread(f"blk.{L}.ffn_gate_inp.weight", 384 * 5120 * 4), np.float32).reshape(384, 5120)
    s = bf16(hf(f"layers.{L}.ffn.gate.weight")[0]).reshape(384, 5120)
    out.append("%.4f" % float(np.mean(g == s)))
    for gname, hname, rows, cols in ((f"blk.{L}.attn_output_b.weight", f"layers.{L}.attn.wo_b.weight", 5120, 8192),
                                     (f"blk.{L}.ffn_down_shexp.weight", f"layers.{L}.ffn.shared_experts.w2.weight", 5120, 2304)):
        G = q8_0(gread(gname, rows * (cols // 32) * 34), rows, cols)
        S = fp8(hname)
        D = (G - S).astype(np.float64)
        rel = float(np.linalg.norm(D) / np.linalg.norm(S))
        sv = top_sv(D)
        worst.append((sv[0] / sv[1], L, gname))
        out.append("%.4f  %s" % (rel, " ".join("%.4f" % x for x in sv)))
    print(" | ".join(out), flush=True)
worst.sort(reverse=True)
print("largest sv1/sv2 ratio of any difference: %.3f (layer %d, %s)" % worst[0], flush=True)
print("DONE", flush=True)
