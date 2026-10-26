#!/usr/bin/env python3
"""qv_score.py WORK - our engine vs the original DeepSeek-V4.1-Flash-1017 (teacher) on the same token rows.
For every engine dump WORK/elogits/{tag}_sNN.logits (f32 [S, V]) with a teacher file WORK/tlogits/sNN.npy, rows
0..S-2 (row t predicts token t+1): KL(teacher || engine), top-1 agreement with the teacher's top token, NLL of the
real next token under both (-> perplexity), per domain and overall.  Writes WORK/qv_score.json."""
import glob, json, os, sys
import numpy as np


def logsm(x):
    m = x.max(1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    W = sys.argv[1]
    meta = json.load(open(os.path.join(W, "meta.json")))
    tokens = np.load(os.path.join(W, "tokens.npy"))
    N, S = tokens.shape
    V = 129280
    acc = {}   # (tag, domain) -> [kl_sum, top1_hits, nll_t, nll_e, rows, kl list]
    per_seq = []
    for path in sorted(glob.glob(os.path.join(W, "elogits", "*_s*.logits"))):
        tag, s = os.path.basename(path)[:-7].rsplit("_s", 1)
        i = int(s)
        tp = os.path.join(W, "tlogits", f"s{i:02d}.npy")
        if not os.path.exists(tp):
            continue
        T = np.load(tp, mmap_mode="r")
        E = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, V)
        assert len(E) >= S - 1 and len(T) >= S - 1, (path, E.shape, T.shape)
        dom = meta["domains"][i]
        kls, hits, nt, ne = [], 0, 0.0, 0.0
        for c in range(0, S - 1, 256):
            r = slice(c, min(c + 256, S - 1))
            lt, le = logsm(np.asarray(T[r], np.float64)), logsm(np.asarray(E[r], np.float64))
            kls.append((np.exp(lt) * (lt - le)).sum(1))
            hits += int((lt.argmax(1) == le.argmax(1)).sum())
            tgt = tokens[i, r.start + 1:r.stop + 1]
            nt -= float(lt[np.arange(len(tgt)), tgt].sum()); ne -= float(le[np.arange(len(tgt)), tgt].sum())
        kl = np.concatenate(kls)
        for key in ((tag, dom), (tag, "ALL")):
            a = acc.setdefault(key, [0.0, 0, 0.0, 0.0, 0, []])
            a[0] += float(kl.sum()); a[1] += hits; a[2] += nt; a[3] += ne; a[4] += len(kl); a[5].append(kl)
        per_seq.append({"tag": tag, "seq": i, "domain": dom, "name": meta["names"][i], "kl": float(kl.mean()),
                        "top1": hits / len(kl), "ppl_teacher": float(np.exp(nt / len(kl))), "ppl_engine": float(np.exp(ne / len(kl)))})
        print(f"{tag:3s} s{i:02d} {dom:5s} KL {kl.mean():.4f}  top1 {hits / len(kl):.3f}  ppl teacher {np.exp(nt / len(kl)):.3f} "
              f"engine {np.exp(ne / len(kl)):.3f}", flush=True)
    out = {}
    for (tag, dom), a in sorted(acc.items()):
        kl = np.concatenate(a[5])
        out[f"{tag}/{dom}"] = {"rows": a[4], "kl_mean": a[0] / a[4], "kl_median": float(np.median(kl)),
                              "kl_p99": float(np.percentile(kl, 99)), "top1_agree": a[1] / a[4],
                              "ppl_teacher": float(np.exp(a[2] / a[4])), "ppl_engine": float(np.exp(a[3] / a[4]))}
    json.dump({"summary": out, "per_seq": per_seq}, open(os.path.join(W, "qv_score.json"), "w"), indent=1)
    print(f"\n{'setting/domain':14s} {'rows':>6s} {'KL mean':>8s} {'KL med':>7s} {'KL p99':>7s} {'top1':>6s} {'ppl T':>7s} {'ppl E':>7s} {'ratio':>6s}")
    for k, v in out.items():
        print(f"{k:14s} {v['rows']:6d} {v['kl_mean']:8.4f} {v['kl_median']:7.4f} {v['kl_p99']:7.3f} {v['top1_agree']:6.3f} "
              f"{v['ppl_teacher']:7.3f} {v['ppl_engine']:7.3f} {v['ppl_engine'] / v['ppl_teacher']:6.3f}")


if __name__ == "__main__":
    main()
