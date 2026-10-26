#!/usr/bin/env python3
"""speedtab.py - llama-bench's speed table for the larry ds41 server, through its OpenAI API: pp512 (prompt tokens/s
reading 512 new tokens) and tg128 (tokens/s generating 128) at context depths 0 / 8K / 32K / 96K, each the mean +-
stddev of R timed runs after untimed warm-up runs, as llama-bench reports them.

The engine keeps a conversation's cache when a request starts with the whole previous prompt (it restores the
snapshot taken at that prompt's end; ds41_generate.cpp "Conversation reuse"), as agentic turns do; it never resumes
in the middle of a prompt.  So depth d is a conversation: a first user message of real text (Xolotl's own sources,
another slice for each depth) read from empty (untimed: its timings are the "prompt from empty" table), then turns
that each add ~512 new tokens (an empty assistant turn + a user turn: a fresh source slice and "Ignore the code
above. Write ... essay") and generate 128.  Each turn is 512 tokens deeper than the last: the first message is sized
so that the R timed turns' depths centre on d.  Depth 0: separate single-turn requests of 512 tokens.  llama-bench
feeds random tokens; this engine's speed depends on the text (the DSpark draft's acceptance, which experts are in
VRAM), so the text is real.  Rates: the server's llama.cpp-style timings, from the engine's clock (pp = prompt_n /
prompt_ms, tg = predicted_n / predicted_ms; cache_n = what the cache held = the turn's depth); with --log, each
request's engine lines too (expert hits by tier, the prompt's phases, DSpark's tokens a pass).

  speedtab.py --port 18081 --label dspark --log logs/engine-dspark.log [--depths 0,8192,32768,98304] [--reps 5]
              [--pp 512] [--tg 128] [--out F]
  speedtab.py --port 18081 --warm 3         warm-up requests only (an essay, code, an essay ...)
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, '/opt/ai/dsv41')
import tok  # noqa: E402  the server's tokenizer (Xolotl's strata_tokenizer, joyai-llm pre-split) and chat encoder

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=18081)
ap.add_argument('--label', default='run')
ap.add_argument('--depths', default='0,8192,32768,98304')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--pp', type=int, default=512)
ap.add_argument('--tg', type=int, default=128)
ap.add_argument('--warm', type=int, default=0)
ap.add_argument('--log', default='')
ap.add_argument('--out', default='')
a = ap.parse_args()
URL = 'http://172.23.0.1:%d/v1/chat/completions' % a.port
KEY = [l.split('=', 1)[1].strip() for l in open('/opt/docker/vllm/.env') if l.startswith('VLLM_API_KEY')][0]
TASK = ('Write a detailed, long technical essay (at least 1500 words) about how virtual memory works in modern '
        'operating systems.')
ESSAY = 'Ignore the code above. ' + TASK
CODE = ('Write a Python module that implements an LRU cache with per-entry TTL expiry: type hints, docstrings, '
        'thread safety, and a full set of unit tests.')
# text the chat encoder would read as special tokens (DeepSeek's markers, think tags, DSML): files with it are skipped
BAD = ('\uff5c', 'think>', 'DSML', '\u2581')
WORDS = ('Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo Lima Mike November Oscar Papa Quebec '
         'Romeo Sierra Tango Uniform Victor Whiskey Xray Yankee Zulu').split()
TK = tok.tokenizer()

# the engine's own lines for one request (stderr -> the server's engine log)
LOG_RE = [
    (re.compile(r'serve: prompt (\d+) tokens \((\d+) reused, ([\w ]+)\) in ([\d.]+) s = ([\d.]+) tok/s \(experts (\d+) '
                r'hit / (\d+) cpu / (\d+) pcie / (\d+) file\)'),
     ('eng_prompt', 'eng_reused', 'eng_mode', 'eng_prompt_s', 'eng_pp', 'pp_hit', 'pp_cpu', 'pp_pcie', 'pp_file')),
    (re.compile(r'serve: prompt phases: .*\(([\d.]+) s: dense ([\d.]+), experts ([\d.]+)'),
     ('ph_s', 'ph_dense', 'ph_experts')),
    (re.compile(r'serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu, (\d+) '
                r'pcie'), ('dec_n', 'dec_s', 'dec_tps', 'dec_finish', 'tg_hit', 'tg_cpu', 'tg_pcie')),
    (re.compile(r'serve: DSpark (\d+) passes, ([\d.]+) tokens/pass'), ('dsp_passes', 'dsp_tpp')),
]


def ntok(s):
    return len(TK.encode(s))


def chat_ids(msgs):
    return TK.encode(tok.E.encode_messages(msgs, thinking_mode='chat'), parse_special=True)


def log_size():
    try:
        return os.path.getsize(a.log) if a.log else 0
    except OSError:
        return 0


def log_since(p0):
    out = {}
    if not a.log:
        return out
    try:
        with open(a.log, 'rb') as f:
            f.seek(p0)
            text = f.read().decode('utf-8', 'ignore')
    except OSError:
        return out
    for l in text.splitlines():
        for rx, keys in LOG_RE:
            m = rx.search(l)
            if m:
                for key, v in zip(keys, m.groups()):
                    out[key] = v if not re.fullmatch(r'[\d.]+', v) else (float(v) if '.' in v else int(v))
    return out


def post(msgs, max_tokens):
    body = {'model': 'larry', 'max_tokens': max_tokens, 'reasoning_effort': 'none', 'messages': msgs}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    p0, t0 = log_size(), time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    tm, u = d.get('timings') or {}, d.get('usage') or {}
    r = {k: tm.get(k) for k in ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_ms',
                                'predicted_per_second')}
    r.update(wall_s=round(time.time() - t0, 2), prompt_tokens=u.get('prompt_tokens'),
             completion_tokens=u.get('completion_tokens'), finish=d['choices'][0].get('finish_reason'))
    time.sleep(0.2)                            # the engine's last lines of this request
    r.update(log_since(p0))
    return r


def line(tag, r):
    extra = ''
    if 'pp_pcie' in r:
        extra += ', prompt experts %s hit / %s pcie / %s cpu' % (r['pp_hit'], r['pp_pcie'], r['pp_cpu'])
    if 'dsp_tpp' in r:
        extra += ', DSpark %.2f tok/pass' % r['dsp_tpp']
    print('%-14s cache %6s new %6s: %7.1f tok/s | gen %4s: %6.1f tok/s (%s)%s, wall %.1f s' % (
        tag, r['cache_n'], r['prompt_n'], r['prompt_per_second'] or 0, r['predicted_n'], r['predicted_per_second'] or 0,
        r['finish'], extra, r['wall_s']), flush=True)


if a.warm:
    for i in range(a.warm):
        line('warm %d' % (i + 1), post([{'role': 'user', 'content': TASK if i % 2 == 0 else CODE}], 256))
    sys.exit(0)


def corpus():
    root = '/opt/ai/dsv41/xolotl'
    files = set()
    for d in ('include', 'src', 'sycl', 'metal', 'tests', 'tools', 'docs', 'cmake'):
        for ext in ('cpp', 'hpp', 'h', 'py', 'md', 'cmake'):
            files.update(glob.glob('%s/%s/**/*.%s' % (root, d, ext), recursive=True))
    out = []
    for f in sorted(files):
        try:
            t = open(f, encoding='utf-8').read().replace('\r', '')
        except (UnicodeDecodeError, OSError):
            continue
        if 200 < len(t) < 200000 and not any(b in t for b in BAD):
            out.append('File %s:\n%s\n' % (os.path.relpath(f, root), t))
    return out


def cut(text, n, line_end):
    """text up to its n-th token (then to the end of that line when line_end), and its token count"""
    ids = TK.encode(text)
    if len(ids) <= n:
        return text, len(ids)
    b = text.encode('utf-8')
    off = sum(len(TK.token_bytes(i)) for i in ids[:n])
    nl = b.find(b'\n', off) if line_end else -1
    t = b[:nl + 1 if nl >= 0 else off].decode('utf-8', 'ignore')
    return t, ntok(t)


files = corpus()
x, y = chat_ids([{'role': 'user', 'content': 'Alpha'}]), chat_ids([{'role': 'user', 'content': 'Bravo'}])
HEAD = next(i for i in range(min(len(x), len(y))) if x[i] != y[i])
TAIL = len(x) - HEAD - ntok('Alpha')
z = chat_ids([{'role': 'user', 'content': 'Alpha'}, {'role': 'assistant', 'content': ''},
              {'role': 'user', 'content': 'Bravo'}])
assert z[:len(x)] == x, 'a turn does not start with the whole previous prompt: no cache reuse'
TURN = len(z) - len(x) - ntok('Bravo')      # the template's tokens a turn adds (end of turn, user, assistant tags)
print('%s: %d source files, %.1f MB; chat template head %d / tail %d / turn %d tokens' % (
    a.label, len(files), sum(map(len, files)) / 1e6, HEAD, TAIL, TURN), flush=True)
pool = ''.join(files[-80:])        # the turns' new text, from the end (the first messages take files from the start)
pos = 0


def new_text(k, n):
    """turn k's new text, n tokens: a word of its own, a fresh source slice, the essay request"""
    global pos
    w = WORDS[k % len(WORDS)] + '. '
    body, _ = cut(pool[pos:pos + 12 * n], n - ntok(w) - ntok('\n' + ESSAY), False)
    pos += len(body)
    return w + body + '\n' + ESSAY


def stats(v):
    v = [x for x in v if x is not None]
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (0.0, 0.0)


res = {'label': a.label, 'pp': a.pp, 'tg': a.tg, 'reps': a.reps, 'turn_tokens': TURN, 'depths': []}
fstart, k = 0, 0
for depth in [int(s) for s in a.depths.split(',')]:
    first, warm, runs = None, [], []
    if depth == 0:
        for r in range(a.reps + 1):
            x = post([{'role': 'user', 'content': new_text(k, a.pp - HEAD - TAIL)}], a.tg)
            k += 1
            line(('warm-up d0' if r == 0 else 'run %d d0' % r), x)
            (runs if r else warm).append(x)
    else:
        # the first message: the R timed turns then read at depth - (R - 1) / 2 * pp ... depth + (R - 1) / 2 * pp
        base_n = depth - (a.reps + 3) * a.pp // 2
        parts, est = [], 0
        while fstart < len(files) - 80 and est < base_n * 1.15:
            parts.append(files[fstart])
            est += len(files[fstart]) / 4.0      # an under-estimate (code is ~3-3.5 characters a token)
            fstart += 1
        base, n = cut(''.join(parts), base_n, True)
        assert n >= base_n * 0.99, (base_n, n)
        msgs = [{'role': 'user', 'content': base + new_text(k, a.pp - TAIL)}]
        k += 1
        first = post(msgs, a.tg)
        line('first d%d' % depth, first)
        for r in range(a.reps + 1):
            msgs += [{'role': 'assistant', 'content': ''}, {'role': 'user', 'content': new_text(k, a.pp - TURN)}]
            k += 1
            x = post(msgs, a.tg)
            line(('warm-up d%d' if r == 0 else 'run %d d%%d' % r) % depth, x)
            (runs if r else warm).append(x)
    pp, tg = stats([x['prompt_per_second'] for x in runs]), stats([x['predicted_per_second'] for x in runs])
    res['depths'].append({'depth': depth, 'first': first, 'warm': warm, 'runs': runs,
                          'pp_mean': pp[0], 'pp_sd': pp[1], 'tg_mean': tg[0], 'tg_sd': tg[1],
                          'cache_n': stats([x['cache_n'] for x in runs])[0],
                          'cache_min': min(x['cache_n'] or 0 for x in runs),
                          'cache_max': max(x['cache_n'] or 0 for x in runs),
                          'prompt_n': stats([x['prompt_n'] for x in runs])[0],
                          'predicted_n': stats([x['predicted_n'] for x in runs])[0],
                          'pp_pcie': stats([x.get('pp_pcie') for x in runs])[0],
                          'dsp_tpp': stats([x.get('dsp_tpp') for x in runs])[0] or None})

print('\n| %s | t/s | new tokens | depth: mean (range) | DSpark tokens/pass |' % a.label)
print('| --- | ---: | ---: | ---: | ---: |')
for d in res['depths']:
    dr = '%.0f (%d-%d)' % (d['cache_n'], d['cache_min'], d['cache_max']) if d['depth'] else '%.0f' % d['cache_n']
    print('| pp%d @ d%d | %.1f ± %.1f | %.0f | %s | |' % (a.pp, d['depth'], d['pp_mean'], d['pp_sd'], d['prompt_n'], dr))
    print('| tg%d @ d%d | %.1f ± %.1f | %.0f | | %s |' % (a.tg, d['depth'], d['tg_mean'], d['tg_sd'], d['predicted_n'],
                                                         '%.2f' % d['dsp_tpp'] if d['dsp_tpp'] else ''))
print('\n| prompt from empty | tokens | s | t/s |')
print('| --- | ---: | ---: | ---: |')
for d in res['depths']:
    f = d['first']
    if f and f['prompt_n']:
        print('| d%d first message | %d | %.1f | %.1f |' % (d['depth'], f['prompt_n'], f['prompt_ms'] / 1000,
                                                            f['prompt_per_second']))
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)
