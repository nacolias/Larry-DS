#!/usr/bin/env python3
"""utprof.py DIR [--gap S] [--top N] - unitrace chrome timelines (logs/ut-NAME/*.json, --chrome-kernel-logging) of
prompts traced one at a time (utdrive.py): the events split into windows at gaps of more than --gap seconds (2), and
per window and card: the wall span, the compute engines' busy time (union of kernel intervals), the copy engines'
busy time, and the top kernels by total time (card 0: --top, 45; cards 1-3: 12), each with its category - what a
prompt's time on card 0 is made of before splitting it.  Streams the JSON a line at a time (traces of ~1 GB)."""
import collections
import glob
import json
import os
import re
import sys

d = sys.argv[1]
gap = float(sys.argv[sys.argv.index('--gap') + 1]) if '--gap' in sys.argv else 2.0
top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 45

CATS = [   # (category, regex on the kernel name) - first match wins
    ('tier', r'strata::v41::xe|xe_tier|run_big|run_small|xe_xmx|xe_dq|planar|hot_'),
    ('tp', r'pc_poll|strata::v41::tp|engine1'),
    ('moe', r'mmid|mul_mat_id'),
    ('copy', r'^zeCommandListAppendMemoryCopy|^zeCommandListAppendMemoryFill'),
    ('gemm', r'gemm|matmul|onednn|dnnl|jit|mul_mat|ref_|gen9|xe_hp|xe2|brgemm|dpas|mmq|mmvq|fastmv|mul_mat_vec'),
    ('dequant', r'dequant|to_fp16|convert_unary|q6_k|q8_0|cpy_f32_f16|cpy_f16|ggml_cpy'),
    ('indexer', r'(?i)li_|lightning|indexer|topk|argsort|top_k'),
    ('attn', r'(?i)dsv4.*attn|attn|flash|softmax|soft_max|kv_fin|kgather|sinks|compress'),
    ('hc', r'dsv4_hc|sinkhorn'),
    ('fq', r'fake_quant'),
    ('rope', r'rope'),
    ('norm', r'rms_norm|norm_'),
    ('elem', r'bin_bcast|scale_|unary|clamp|sigmoid|swiglu|softplus|sqrt|concat|get_rows|set_rows|pad_|sum_rows|repeat|'
             r'cont|cpy|argmax|exp|log|mul|add'),
]
CRX = [(c, re.compile(r)) for c, r in CATS]


def cat(name):
    head = name if name.startswith('zeCommandList') else name.split('(')[0]
    for c, rx in CRX:
        if rx.search(head):
            return c
    return 'other'


def short(k):
    s = k.replace('sycl::_V1::', '').replace('(anonymous namespace)::', '').replace('ggml_backend_sycl_context&', 'ctx')
    for _ in range(4):
        s = re.sub(r'\((?:[\w:<>&*, ]|const)*\)', '(.)', s)
    return s[:120]


devname, tname = {}, {}
evs = []   # (ts, end, pid, tid, key)
files = sorted(glob.glob(os.path.join(d, '*.json')))
if not files:
    sys.exit('no json in ' + d)
for f in files:
    with open(f) as fh:
        for ln in fh:
            ln = ln.strip().rstrip(',')
            if not ln.startswith('{"ph"'):
                continue
            if ln.startswith('{"ph": "M"'):
                e = json.loads(ln)
                if e['name'] == 'process_name':
                    devname[e['pid']] = e['args']['name']
                elif e['name'] == 'thread_name':
                    tname[(e['pid'], e.get('tid'))] = e['args']['name']
                continue
            if not ln.startswith('{"ph": "X"'):
                continue
            e = json.loads(ln)
            if 'dur' not in e:
                continue
            k = e['name'].split(' [SIMD')[0]
            evs.append((e['ts'], e['ts'] + e['dur'], e['pid'], e.get('tid'), k))


def card(pid):
    m = re.search(r'#(\d+)$', devname.get(pid, ''))
    return int(m.group(1)) if m else None


evs = [e for e in evs if card(e[2]) is not None]   # device lanes only (the host lane: API calls)
evs.sort()
print('%d device events in %s' % (len(evs), ', '.join(os.path.basename(f) for f in files)))
wins, cur, last = [], [], None
for e in evs:
    if last is not None and e[0] - last > gap * 1e6:
        wins.append(cur)
        cur = []
    cur.append(e)
    last = max(last or 0, e[1])
if cur:
    wins.append(cur)


def busy(iv):
    iv.sort()
    t, s, e = 0.0, None, None
    for a, b in iv:
        if s is None or a > e:
            if s is not None:
                t += e - s
            s, e = a, b
        else:
            e = max(e, b)
    if s is not None:
        t += e - s
    return t


for wi, w in enumerate(wins):
    t0, t1 = w[0][0], max(e[1] for e in w)
    print('\n=== window %d: %.2f s wall, %d events' % (wi, (t1 - t0) / 1e6, len(w)))
    for c in sorted({card(e[2]) for e in w}):
        ce = [e for e in w if card(e[2]) == c]
        comp = [(e[0], e[1]) for e in ce if 'Copy' not in tname.get((e[2], e[3]), '') and not e[4].startswith('zeCommandList')]
        copy = [(e[0], e[1]) for e in ce if 'Copy' in tname.get((e[2], e[3]), '') or e[4].startswith('zeCommandList')]
        agg = collections.defaultdict(lambda: [0, 0.0])
        cagg = collections.defaultdict(float)
        for e in ce:
            a = agg[e[4]]
            a[0] += 1
            a[1] += e[1] - e[0]
            cagg[cat(e[4])] += e[1] - e[0]
        print('--- card %d: compute busy %.3f s, copy busy %.3f s (span %.2f s); by category (summed kernel time): %s' % (
            c, busy(comp) / 1e6, busy(copy) / 1e6, (max(e[1] for e in ce) - min(e[0] for e in ce)) / 1e6,
            ', '.join('%s %.3f' % (k, v / 1e6) for k, v in sorted(cagg.items(), key=lambda kv: -kv[1]))))
        for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top if c == 0 else 12]:
            print('  %8.1f ms %7d x %8.1f us  %-7s %s' % (t / 1e3, n, t / n, cat(k), short(k)))
