#!/usr/bin/env python3
"""qv_head.py OUTDIR - the teacher's logits: v41_calib.py cmd_head's math (last pre-mix hc_pre, final norm, fp32
head on h/L40) for every row of every sequence, written as OUTDIR/sNN.logits (f32 [S, V], row t predicts token t+1 -
the engine's --dump-logits layout) plus the per-domain NLL / top-1 v41_calib prints."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/opt/ai/dsv41/xolotl/tools/v41")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import v41_calib as C  # noqa: E402


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    from safetensors import safe_open
    meta = json.load(open(C.wpath("meta.json")))
    tokens = np.load(C.wpath("tokens.npy"))
    N, S = tokens.shape
    torch.set_default_dtype(C.BF16)
    args = C.model_args(1, S)
    wm = C.weight_map()
    norm = C.M.RMSNorm(C.D, args.norm_eps)
    with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(C.shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(torch.float32)
    h_in = C.h_memmap(40, "r", N, S)
    pre = np.load(C.wpath("pre", "L40.npy"))
    res = {}
    with torch.inference_mode():
        for i in range(N):
            x = torch.from_numpy(np.asarray(h_in[i:i + 1])).view(C.BF16)
            pm = torch.from_numpy(pre[i:i + 1])
            y = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]     # [S, D]
            lg = np.lib.format.open_memmap(os.path.join(out, f"s{i:02d}.npy"), mode="w+", dtype=np.float32, shape=(S, W.shape[0]))
            tgt = torch.from_numpy(tokens[i, 1:].astype(np.int64))
            nll, hit = 0.0, 0
            for c in range(0, S, 512):
                l = y[c:c + 512].float() @ W.T
                lg[c:c + l.shape[0]] = l.numpy()
                n = min(l.shape[0], S - 1 - c)
                if n > 0:
                    nll += float(F.cross_entropy(l[:n], tgt[c:c + n], reduction="sum"))
                    hit += int((l[:n].argmax(-1) == tgt[c:c + n]).sum())
            lg.flush(); del lg
            r = res.setdefault(meta["domains"][i], [0.0, 0, 0])
            r[0] += nll; r[1] += hit; r[2] += S - 1
            C.log(f"head: s{i:02d} {meta['domains'][i]} ppl {np.exp(nll / (S - 1)):.4f} top1 {hit / (S - 1):.3f}")
    tot = [sum(v[j] for v in res.values()) for j in range(3)]
    res["ALL"] = tot
    summary = {d: {"ppl": float(np.exp(v[0] / v[2])), "top1": v[1] / v[2], "tokens": v[2]} for d, v in res.items()}
    json.dump(summary, open(os.path.join(out, "teacher_summary.json"), "w"), indent=1)
    for d, v in summary.items():
        C.log(f"head: {d:6s} ppl {v['ppl']:.4f} top1 {v['top1']:.3f} ({v['tokens']} tokens)")


if __name__ == "__main__":
    main()
