#!/usr/bin/env python3
"""abx.py --label NAME [--prompts 3000,4400,6000,8000] [--prompt-reps 2] [--edit] [--dec] [--log ENGINE_LOG]
[--out F] - the 2026-10-24 afternoon A/B through the ds41 server's OpenAI API (port 18081, greedy, no thinking):
  --prompts  first messages read from empty (no reuse: each its own word + source slice) of about N tokens, 16 out:
             the prompt path's time for new conversations (STRATA_XE_PREFETCH_DEPTH)
  --edit     an agentic edit: a source file of ~9-11K tokens and "reproduce it with a comment before every function",
             1200 out, run twice (the second a regen: decode only) - lookup drafts (STRATA_LOOKUP_TP)
  --dec      an essay, code and a math solution, 400 out each - DSpark's passes and the CPU pool's experts a pass
Each request's engine lines (prompt phases, decode, DSpark passes, lookup) come from --log.  Rows go to --out (JSON
lines) and a summary line per request is printed.  Run with /opt/ai/dsv41/venv/bin/python -I."""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, '/opt/ai/dsv41')
import tok  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=18081)
ap.add_argument('--label', required=True)
ap.add_argument('--prompts', default='')
ap.add_argument('--prompt-reps', type=int, default=2)
ap.add_argument('--edit', action='store_true')
ap.add_argument('--dec', action='store_true')
ap.add_argument('--needles', default='', help='long-context needles of about N tokens from empty, comma-separated')
ap.add_argument('--log', default='')
ap.add_argument('--out', default='')
ap.add_argument('--stop-flag', default='', help='stop between requests when this file exists')
a = ap.parse_args()
URL = 'http://172.23.0.1:%d/v1/chat/completions' % a.port
KEY = [ln.split('=', 1)[1].strip() for ln in open('/opt/docker/vllm/.env') if ln.startswith('VLLM_API_KEY')][0]
TK = tok.tokenizer()
BAD = ('\uff5c', 'think>', 'DSML', '\u2581')
WORDS = ('Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo Lima Mike November Oscar Papa Quebec '
         'Romeo Sierra Tango Uniform Victor Whiskey Xray Yankee Zulu').split()
LOG_RE = [
    (re.compile(r'serve: prompt (\d+) tokens \((\d+) reused, ([\w ]+)\) in ([\d.]+) s = ([\d.]+) tok/s \(experts (\d+) '
                r'hit / (\d+) cpu / (\d+) pcie'),
     ('eng_prompt', 'eng_reused', 'eng_mode', 'eng_prompt_s', 'eng_pp', 'pp_hit', 'pp_cpu', 'pp_pcie')),
    (re.compile(r'serve: prompt phases: (\d+) new = (\d+) batch \(([\d.]+) s: dense ([\d.]+), experts ([\d.]+).*cache open '
                r'([\d.]+) s, seed ([\d.]+) s'),
     ('ph_new', 'ph_batch', 'ph_s', 'ph_dense', 'ph_experts', 'ph_open', 'ph_seed')),
    (re.compile(r'serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu'),
     ('dec_n', 'dec_s', 'dec_tps', 'dec_finish', 'tg_hit', 'tg_cpu')),
    (re.compile(r'serve: DSpark (\d+) passes, ([\d.]+) tokens/pass'), ('dsp_passes', 'dsp_tpp')),
    (re.compile(r'serve: lookup (\d+)/(\d+) drafts accepted in (\d+) passes'), ('lk_acc', 'lk_dr', 'lk_passes')),
]


def ntok(s):
    return len(TK.encode(s))


def log_size():
    try:
        return os.path.getsize(a.log) if a.log else 0
    except OSError:
        return 0


def log_since(p0):
    out = {}
    if not a.log:
        return out
    with open(a.log, 'rb') as f:
        f.seek(p0)
        text = f.read().decode('utf-8', 'ignore')
    for ln in text.splitlines():
        for rx, keys in LOG_RE:
            m = rx.search(ln)
            if m:
                for key, v in zip(keys, m.groups()):
                    out[key] = v if not re.fullmatch(r'[\d.]+', v) else (float(v) if '.' in v else int(v))
    return out


def post(msgs, max_tokens):
    body = {'model': 'larry', 'max_tokens': max_tokens, 'reasoning_effort': 'none', 'temperature': 0.0,
            'messages': msgs}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    p0, t0 = log_size(), time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    u = d.get('usage') or {}
    r = {'wall_s': round(time.time() - t0, 2), 'prompt_tokens': u.get('prompt_tokens'),
         'completion_tokens': u.get('completion_tokens'), 'finish': d['choices'][0].get('finish_reason'),
         'text_head': (d['choices'][0]['message'].get('content') or '')[:120]}
    time.sleep(0.3)
    r.update(log_since(p0))
    return r


def corpus():
    root = '/opt/ai/dsv41/xolotl'
    files = set()
    for d in ('include', 'src', 'sycl', 'tests', 'tools', 'docs', 'cmake'):
        for ext in ('cpp', 'hpp', 'h', 'py', 'md'):
            files.update(glob.glob('%s/%s/**/*.%s' % (root, d, ext), recursive=True))
    out = []
    for f in sorted(files):
        try:
            t = open(f, encoding='utf-8').read().replace('\r', '')
        except (UnicodeDecodeError, OSError):
            continue
        if 200 < len(t) < 400000 and not any(b in t for b in BAD):
            out.append((os.path.relpath(f, root), t))
    return out


def cut(text, n):
    ids = TK.encode(text)
    if len(ids) <= n:
        return text
    b = text.encode('utf-8')
    off = sum(len(TK.token_bytes(i)) for i in ids[:n])
    nl = b.find(b'\n', off)
    return b[:nl + 1 if nl >= 0 else off].decode('utf-8', 'ignore')


def show(tag, r):
    s = '%-22s' % tag
    if 'eng_prompt_s' in r:
        s += ' prompt %6s (%s reused) %6.2f s = %6.1f tok/s' % (r['eng_prompt'], r['eng_reused'], r['eng_prompt_s'],
                                                               r['eng_pp'])
        if 'ph_dense' in r:
            s += ' [dense %.2f experts %.2f open %.2f seed %.2f]' % (r['ph_dense'], r['ph_experts'], r['ph_open'],
                                                                     r['ph_seed'])
    if 'dec_n' in r:
        s += ' | dec %4d in %6.2f s = %5.1f tok/s' % (r['dec_n'], r['dec_s'], r['dec_tps'])
        if r.get('dsp_passes'):
            s += ', %d passes %.2f tok/p %.1f ms/p, cpu %.1f/p' % (r['dsp_passes'], r['dsp_tpp'],
                                                                 1e3 * r['dec_s'] / r['dsp_passes'],
                                                                 r['tg_cpu'] / r['dsp_passes'])
        if 'lk_passes' in r:
            s += ', lookup %d/%d in %d' % (r['lk_acc'], r['lk_dr'], r['lk_passes'])
    print(s, flush=True)


rows = []


def rec(kind, tag, r):
    r.update(label=a.label, kind=kind, tag=tag)
    rows.append(r)
    show(tag, r)
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(r) + '\n')


def stop():
    return bool(a.stop_flag) and os.path.exists(a.stop_flag)


files = corpus()
if a.prompts:
    pool = ''.join('File %s:\n%s\n' % ft for ft in files)
    pos = 0
    k = 0
    for n in [int(x) for x in a.prompts.split(',')]:
        for rep in range(a.prompt_reps):
            if stop():
                sys.exit(3)
            w = '%s %d. ' % (WORDS[k % len(WORDS)], k)
            k += 1
            ask = '\nSummarize the code above in one sentence.'
            body = cut(pool[pos:pos + 14 * n], n - ntok(w) - ntok(ask))
            pos += len(body)
            rec('prompt', 'p%d.%d' % (n, rep), post([{'role': 'user', 'content': w + body + ask}], 16))
if a.edit:
    cand = [(p, t) for p, t in files if p.endswith(('.cpp', '.py', '.hpp')) and 25000 <= len(t) <= 50000]
    cand = sorted(((p, t) for p, t in cand if 9000 <= ntok(t) <= 11000), key=lambda pt: not pt[0].endswith('.cpp'))
    if not cand:
        sys.exit('no source file of 9-11K tokens')
    path, text = cand[0]
    msg = [{'role': 'user', 'content': 'File %s:\n```\n%s```\n\nReproduce this file exactly, but insert a one-line '
                                       'comment `// reviewed` (or `# reviewed` in Python) on the line before every '
                                       'function definition. Output only the file.' % (path, text)}]
    for rep in range(2):
        if stop():
            sys.exit(3)
        rec('edit', 'edit.%d %s' % (rep, os.path.basename(path)), post(msg, 1200))
if a.needles:   # needle.py's prompt: a passphrase in the middle of N tokens of the engine's own C++ sources
    src = ''.join(open(f).read() for f in sorted(glob.glob('/opt/ai/dsv41/xolotl/tools/v41/engine/*.cpp')))
    import random
    for n in [int(x) for x in a.needles.split(',')]:
        if stop():
            sys.exit(3)
        chars = n * 3
        filler = (src * (chars // len(src) + 1))[:chars]
        mid = len(filler) // 2
        needle = "\n// NOTE FOR REVIEWERS: the deployment passphrase is 'violet-kestrel-4217'.\n"
        q = ("Session %08x. Here is a large source dump:\n\n" % random.getrandbits(32) + filler[:mid] + needle + filler[mid:] +
             "\n\nSomewhere in it is a deployment passphrase. What is it? Answer with the passphrase only.")
        msgs = [{'role': 'user', 'content': q}]
        r = post(msgs, 40)
        r['found'] = 'violet-kestrel-4217' in r.get('text_head', '')
        rec('needle', 'needle%d' % n, r)
        print('   needle %d: %s' % (n, 'FOUND' if r['found'] else 'MISSED ' + repr(r.get('text_head', ''))), flush=True)
        if stop():
            sys.exit(3)
        # decode at this depth: the same conversation goes on (its prompt reused), ~400 tokens out
        msgs = msgs + [{'role': 'assistant', 'content': r.get('text_head', '')},
                       {'role': 'user', 'content': 'Now explain what the code in the dump does, in about 300 words.'}]
        r2 = post(msgs, 400)
        rec('needle', 'deep%d' % n, r2)
        if stop():
            sys.exit(3)
        # a big turn at this depth after the decode (the TP core's KV rows are mapped by now): ~12K new tokens
        more = (src * 2)[:36000]
        msgs = msgs + [{'role': 'assistant', 'content': r2.get('text_head', '')},
                       {'role': 'user', 'content': 'Here is more of the source:\n\n' + more + '\n\nIn one sentence: what does '
                                                   'this part do?'}]
        rec('needle', 'turn%d' % n, post(msgs, 60))
if a.dec:
    tasks = [('essay', 'Write a detailed, long technical essay (at least 1500 words) about how virtual memory works in '
                       'modern operating systems.'),
             ('code', 'Write a Python module that implements an LRU cache with per-entry TTL expiry: type hints, '
                      'docstrings, thread safety, and a full set of unit tests.'),
             ('math', 'Solve step by step, showing all work: find all real x with x^4 - 10x^3 + 35x^2 - 50x + 24 = 0, '
                      'then compute the sum of the squares of the roots and verify it with Vieta\'s formulas.')]
    for name, q in tasks:
        if stop():
            sys.exit(3)
        rec('dec', 'dec.' + name, post([{'role': 'user', 'content': q}], 400))
