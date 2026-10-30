#!/usr/bin/env python3
"""agentd.py --label NAME [--port 18081] [--first 64000] [--turns 24] [--max-out 4096] [--log ENGINE_LOG] [--out F] -
a deep agentic session replayed through a ds41 server with thinking on and the service's sampling (temperature 1,
top_p 0.95, a fixed seed a turn), shaped like the 2026-10-29 agent session (136 follow-ups: new tokens p25 71, p50 255,
p75 559, p90 1,155; replies p50 350, mean 862 tokens): a first prompt of --first tokens (the project's docs, then earlier
tool results of its code), then turns that add a fixed assistant tool call and a tool result.  The content is this
machine's own source tree (xolotl), never a user's prompts; the assistant turns in the history are fixed texts, so every
config sees the same prompts.  Per turn the engine's prompt time, decode tok/s, DSpark tokens a pass, ms a pass and
CPU-pool experts a pass (the engine log's lines, as agentx.py).  Rows to --out (JSON lines).  Run with
/opt/ai/dsv41/venv/bin/python -I."""
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
ap.add_argument('--log', default='')
ap.add_argument('--out', default='')
ap.add_argument('--first', type=int, default=64000)
ap.add_argument('--turns', type=int, default=24)
ap.add_argument('--max-out', type=int, default=4096)
ap.add_argument('--temp', type=float, default=1.0, help='temperature (top_p stays 0.95)')
ap.add_argument('--stop-flag', default='', help='stop between requests when this file exists')
a = ap.parse_args()
URL = 'http://172.23.0.1:%d/v1/chat/completions' % a.port
KEY = [ln.split('=', 1)[1].strip() for ln in open('/opt/docker/vllm/.env') if ln.startswith('VLLM_API_KEY')][0]
TK = tok.tokenizer()
BAD = ('\uff5c', 'think>', 'DSML', '\u2581')
LOG_RE = [
    (re.compile(r'serve: prompt (\d+) tokens \((\d+) reused, ([\w ]+)\) in ([\d.]+) s = ([\d.]+) tok/s \(experts (\d+) '
                r'hit / (\d+) cpu / (\d+) pcie'),
     ('eng_prompt', 'eng_reused', 'eng_mode', 'eng_prompt_s', 'eng_pp', 'pp_hit', 'pp_cpu', 'pp_pcie')),
    (re.compile(r'serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu'),
     ('dec_n', 'dec_s', 'dec_tps', 'dec_finish', 'tg_hit', 'tg_cpu')),
    (re.compile(r'serve: DSpark (\d+) passes, ([\d.]+) tokens/pass'), ('dsp_passes', 'dsp_tpp')),
    (re.compile(r'serve: lookup (\d+)/(\d+) drafts accepted in (\d+) passes'), ('lk_acc', 'lk_dr', 'lk_passes')),
]
# new tokens a turn, drawn to the session's quantiles (p10 31 .. p90 1,155, a few larger reads)
SIZES = [255, 71, 559, 31, 1155, 255, 400, 120, 2400, 300, 60, 800, 255, 150, 1500, 500, 90, 650, 255, 3000, 200, 450,
         100, 900]
CALLS = ['{"tool": "read_file", "args": {"path": "%s", "start": 1}}', '{"tool": "grep", "args": {"pattern": "pass_v", '
         '"path": "%s"}}', '{"tool": "run", "args": {"cmd": "sed -n 1,200p %s", "timeout_s": 30}}']


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


def post(msgs, turn):
    body = {'model': 'larry', 'max_tokens': a.max_out, 'temperature': a.temp, 'top_p': 0.95, 'seed': 1000 + turn,
            'messages': msgs}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    p0, t0 = log_size(), time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    u = d.get('usage') or {}
    m = d['choices'][0]['message']
    r = {'wall_s': round(time.time() - t0, 2), 'prompt_tokens': u.get('prompt_tokens'),
         'completion_tokens': u.get('completion_tokens'), 'finish': d['choices'][0].get('finish_reason'),
         'thinking_chars': len(m.get('reasoning_content') or ''), 'answer_chars': len(m.get('content') or '')}
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


class Feed:
    """Source text in order, handed out in slices of about n tokens (as `sed -n` tool output)."""

    def __init__(self, files, order):
        self.parts = [(p, t) for p, t in files if order(p)]
        self.i, self.off, self.last = 0, 0, ''

    def take(self, n):
        out, got = [], 0
        while got < n and self.i < len(self.parts):
            p, t = self.parts[self.i]
            body = cut(t[self.off:self.off + 8 * (n - got) + 64], n - got)
            if not body:
                if n - got < 40:
                    break
                self.i, self.off = self.i + 1, 0
                continue
            lo = t.count('\n', 0, self.off) + 1
            out.append('$ sed -n %d,%dp %s\n%s' % (lo, lo + body.count('\n'), p, body))
            self.last = p
            got += ntok(out[-1]) + 4
            self.off += len(body)
            if self.off >= len(t) - 1:
                self.i, self.off = self.i + 1, 0
        return '\n'.join(out)


def stop():
    return bool(a.stop_flag) and os.path.exists(a.stop_flag)


def show(tag, r):
    s = '%-6s prompt %6s (%6s reused) %6.2f s' % (tag, r.get('eng_prompt'), r.get('eng_reused'), r.get('eng_prompt_s', 0))
    if 'dec_n' in r:
        s += ' | dec %4d in %6.2f s = %5.1f tok/s' % (r['dec_n'], r['dec_s'], r['dec_tps'])
        if r.get('dsp_passes'):
            s += ', %.2f tok/p %.1f ms/p cpu %.1f/p' % (r['dsp_tpp'], 1e3 * r['dec_s'] / r['dsp_passes'],
                                                        r['tg_cpu'] / r['dsp_passes'])
    s += ' [%s]' % r.get('finish')
    print(s, flush=True)


files = corpus()
code = Feed(files, lambda p: p.startswith('tools/v41/') and p.endswith(('.cpp', '.hpp', '.py')))
early = Feed(files, lambda p: p.startswith(('src/', 'include/')) and p.endswith(('.cpp', '.h', '.hpp')))
docs = ''.join('File %s:\n%s\n' % ft for ft in files if ft[0].endswith('.md'))
TOOLS = '''# Tools
You can call these tools by replying with one JSON object on its own line: {"tool": NAME, "args": {...}}.
- read_file {"path": str, "start": int, "end": int}: lines of a file in the repository.
- grep {"pattern": str, "path": str}: matching lines with file names and line numbers.
- edit_file {"path": str, "old": str, "new": str}: replace an exact, unique snippet.
- run {"cmd": str, "timeout_s": int}: a shell command in the repository root; stdout and stderr come back.
Tool results arrive in the next user message.  Never invent tool output.
'''
sys_prompt = ('You are a coding agent working in a C++/SYCL inference engine repository (Xolotl: DeepSeek-V4.1 on '
              'Intel Arc GPUs).  Work step by step: read code with the tools, think about what you find, and make '
              'careful, minimal edits.\n\n' + TOOLS + '\n# Project documentation\n')
sys_prompt += cut(docs, min(a.first // 2, ntok(docs)))
task = ('Task: review the DeepSeek-V4.1 engine under tools/v41/engine for correctness problems in how a serve request '
        'is turned into prompt batches and decode passes.  The files you already read are below.  Continue with the '
        'tools: report each problem with the code involved and propose a fix as an edit.\n\n# Files read so far\n')
task += early.take(a.first - ntok(sys_prompt) - ntok(task) - 40)
msgs = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': task}]
r = post(msgs, 0)
r.update(label=a.label, tag='t0')
show('t0', r)
rows = [r]
for turn in range(1, a.turns + 1):
    if stop():
        sys.exit(3)
    n = SIZES[(turn - 1) % len(SIZES)]
    result = code.take(max(16, n - 60))
    call = CALLS[turn % len(CALLS)] % (code.last or 'tools/v41/engine/ds41_generate.cpp')
    msgs = msgs + [{'role': 'assistant', 'content': call},
                   {'role': 'user', 'content': 'Tool result:\n' + result + '\nContinue the review.'}]
    r = post(msgs, turn)
    r.update(label=a.label, tag='t%d' % turn)
    show('t%d' % turn, r)
    rows.append(r)
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
d = [x for x in rows[1:] if x.get('dsp_passes')]
if d:
    P = sum(x['dsp_passes'] for x in d)
    T = sum(x['dec_n'] for x in d)
    S = sum(x['dec_s'] for x in d)
    C = sum(x['tg_cpu'] for x in d)
    print('turns 1-%d: decode %d tokens in %.1f s = %.1f tok/s, %.2f tok/pass, %.1f ms/pass, cpu %.1f/pass; '
          'prompts %.1f s' % (len(rows) - 1, T, S, T / S, T / P, 1e3 * S / P, C / P,
                              sum(x.get('eng_prompt_s', 0) for x in rows[1:])), flush=True)
