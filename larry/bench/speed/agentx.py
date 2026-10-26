#!/usr/bin/env python3
"""agentx.py --label NAME [--log ENGINE_LOG] [--out F] [--turns N] [--no-side] - an agentic coding session replayed
through the ds41 server's OpenAI API (port 18081, greedy, no thinking), shaped like a real one (2026-10-24 14:13: a
7.8K-token first prompt, then tool results of 17-6,500 tokens with the context growing to ~43K, two side requests of
another conversation in between).  The content is this machine's own source tree (xolotl), never a user's prompts.
The assistant turns in the history are FIXED texts (not the model's replies), so every config sees the same prompts:
each turn's new tokens = the fixed reply + the tool result, read on top of the last prompt's snapshot.  Per turn: the
engine's prompt time and phases, decode tok/s, DSpark passes, CPU-pool experts a pass.  Rows to --out (JSON lines).
Run with /opt/ai/dsv41/venv/bin/python -I."""
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
ap.add_argument('--turns', type=int, default=99)
ap.add_argument('--no-side', action='store_true', help='leave out the side requests')
ap.add_argument('--no-final', action='store_true', help='leave out the last long answer')
ap.add_argument('--dec-scale', type=float, default=1.0, help='scale every turn\'s max tokens')
ap.add_argument('--stop-flag', default='', help='stop between requests when this file exists')
ap.add_argument('--dry', action='store_true', help='print each request\'s prompt tokens, post nothing')
a = ap.parse_args()
URL = 'http://172.23.0.1:%d/v1/chat/completions' % a.port
KEY = [ln.split('=', 1)[1].strip() for ln in open('/opt/docker/vllm/.env') if ln.startswith('VLLM_API_KEY')][0]
TK = tok.tokenizer()
BAD = ('\uff5c', 'think>', 'DSML', '\u2581')
LOG_RE = [
    (re.compile(r'serve: prompt (\d+) tokens \((\d+) reused, ([\w ]+)\) in ([\d.]+) s = ([\d.]+) tok/s \(experts (\d+) '
                r'hit / (\d+) cpu / (\d+) pcie'),
     ('eng_prompt', 'eng_reused', 'eng_mode', 'eng_prompt_s', 'eng_pp', 'pp_hit', 'pp_cpu', 'pp_pcie')),
    (re.compile(r'serve: prompt phases: (\d+) new = (\d+) batch \(([\d.]+) s: dense ([\d.]+), experts ([\d.]+).*DSpark tail '
                r'(\d+) (from the batch|as passes) \(cache open ([\d.]+) s, seed ([\d.]+) s'),
     ('ph_new', 'ph_batch', 'ph_s', 'ph_dense', 'ph_experts', 'ph_tail', 'ph_tailmode', 'ph_open', 'ph_seed')),
    (re.compile(r'serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu'),
     ('dec_n', 'dec_s', 'dec_tps', 'dec_finish', 'tg_hit', 'tg_cpu')),
    (re.compile(r'serve: DSpark (\d+) passes, ([\d.]+) tokens/pass'), ('dsp_passes', 'dsp_tpp')),
    (re.compile(r'serve: lookup (\d+)/(\d+) drafts accepted in (\d+) passes'), ('lk_acc', 'lk_dr', 'lk_passes')),
    (re.compile(r'dense card 0 regrown: (\d+) experts .* in ([\d.]+) s'), ('rg_n', 'rg_s')),
]

# the session's shape: (new tokens, max tokens out); 'side' = a request of another conversation from empty
SHAPE = [(7788, 102), (101, 204), (17, 358), (3664, 255), (100, 161), ('side', 14948, 300), (6501, 351), (655, 178),
         (3683, 279), (747, 217), (2199, 357), (52, 357), (1703, 631), (1409, 290), (1474, 249), ('side', 22239, 300),
         (3626, 281), (3828, 756), (72, 52), (491, 507), (124, 1384), ('final', 1500)]

REPLIES = [
    'I will start by reading the engine\'s main loop to see how a request flows through it.',
    'Next I need the tier\'s header to check the call contract.',
    'Let me look at how the batch path splits its work.',
    'That part looks fine. I will grep for the other call sites.',
    'Now the build script, to see which flags apply.',
    'Let me read the next section of that file.',
    'I want to compare this with the kernel that consumes it.',
    'The sizes must match the planner; checking the planner next.',
    'One more file: the test that covers this path.',
    'Let me check the log output from the last run.',
    'I will open the helper this calls into.',
    'Now the remaining part of the file.',
    'Let me check the configuration it reads at startup.',
    'Reading the caller once more before I edit.',
    'Let me see the diff of the related commit.',
    'Checking the tests again after that change.',
    'Last check: the error path.',
    'Let me run the build to see if it compiles.',
    'And the test output.',
    'One last look at the summary.',
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
    if a.dry:
        return {'eng_prompt': sum(ntok(m['content']) for m in msgs), 'eng_reused': 0, 'eng_prompt_s': 0.0}
    body = {'model': 'larry', 'max_tokens': max_tokens, 'reasoning_effort': 'none', 'temperature': 0.0,
            'messages': msgs}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    p0, t0 = log_size(), time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    u = d.get('usage') or {}
    r = {'wall_s': round(time.time() - t0, 2), 'prompt_tokens': u.get('prompt_tokens'),
         'completion_tokens': u.get('completion_tokens'), 'finish': d['choices'][0].get('finish_reason'),
         'text_head': (d['choices'][0]['message'].get('content') or '')[:200]}
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
    """Source text in order (code files, then docs), handed out in slices of about n tokens."""

    def __init__(self, files, order):
        self.parts = [(p, t) for p, t in files if order(p)]
        self.i, self.off = 0, 0

    def take(self, n):
        out, got = [], 0
        while got < n and self.i < len(self.parts):
            p, t = self.parts[self.i]
            body = cut(t[self.off:self.off + 8 * (n - got) + 64], n - got)
            if not body:
                if n - got < 200:
                    break
                self.i, self.off = self.i + 1, 0
                continue
            lo = t.count('\n', 0, self.off) + 1
            out.append('$ sed -n %d,%dp %s\n%s' % (lo, lo + body.count('\n'), p, body))
            got += ntok(out[-1]) + 4
            self.off += len(body)
            if self.off >= len(t) - 1:
                self.i, self.off = self.i + 1, 0
        return '\n'.join(out)


TOOLS = '''# Tools
You can call these tools by replying with one JSON object on its own line: {"tool": NAME, "args": {...}}.
- read_file {"path": str, "start": int, "end": int}: lines of a file in the repository.
- grep {"pattern": str, "path": str}: matching lines with file names and line numbers.
- edit_file {"path": str, "old": str, "new": str}: replace an exact, unique snippet.
- run {"cmd": str, "timeout_s": int}: a shell command in the repository root; stdout and stderr come back.
- list_dir {"path": str}: the entries of a directory.
Tool results arrive in the next user message.  Never invent tool output.
'''


def stop():
    return bool(a.stop_flag) and os.path.exists(a.stop_flag)


def show(tag, r):
    s = '%-6s' % tag
    if 'eng_prompt_s' in r:
        s += ' prompt %6s (%6s reused) %6.2f s' % (r['eng_prompt'], r['eng_reused'], r['eng_prompt_s'])
        if 'ph_dense' in r:
            s += ' [%s: dense %.2f experts %.2f open %.2f seed %.2f]' % (
                'batch' if r['ph_batch'] else 'passes', r['ph_dense'], r['ph_experts'], r['ph_open'], r['ph_seed'])
    if 'dec_n' in r:
        s += ' | dec %4d in %6.2f s = %5.1f tok/s' % (r['dec_n'], r['dec_s'], r['dec_tps'])
        if r.get('dsp_passes'):
            s += ', %.2f tok/p %.1f ms/p cpu %.1f/p' % (r['dsp_tpp'], 1e3 * r['dec_s'] / r['dsp_passes'],
                                                        r['tg_cpu'] / r['dsp_passes'])
    print(s, flush=True)


def rec(tag, kind, r):
    r.update(label=a.label, kind=kind, tag=tag)
    show(tag, r)
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
    return r


files = corpus()
code = Feed(files, lambda p: p.startswith('tools/v41/') and p.endswith(('.cpp', '.hpp', '.py')))
side_feed = Feed(files, lambda p: p.startswith(('src/', 'include/')) and p.endswith(('.cpp', '.h', '.hpp')))
docs = ''.join('File %s:\n%s\n' % ft for ft in files if ft[0].endswith('.md'))
sys_prompt = ('You are a coding agent working in a C++/SYCL inference engine repository (Xolotl: DeepSeek-V4.1 on '
              'Intel Arc GPUs).  Work step by step: read code with the tools, explain what you find, and make careful, '
              'minimal edits.  Quote the code you change.\n\n' + TOOLS + '\n# Project documentation\n')
task = ('Task: review the DeepSeek-V4.1 engine under tools/v41/engine for correctness problems in how a serve request '
        'is turned into prompt batches and decode passes.  Read the relevant files, report each problem with the code '
        'involved, and propose a fix as a code edit.')
first_n = SHAPE[0][0]
sys_prompt += cut(docs, first_n - ntok(sys_prompt) - ntok(task) - 40)
msgs = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': task}]
rows = []
turn = 0
for step in SHAPE:
    if stop():
        sys.exit(3)
    if step[0] == 'side':
        if a.no_side:
            continue
        _, n, mx = step
        q = 'Summarize what the following code does, file by file, in a short list.'
        body = side_feed.take(n - ntok(q) - 30)
        rows.append(rec('side', 'side', post([{'role': 'user', 'content': body + '\n\n' + q}],
                                             max(1, int(mx * a.dec_scale)))))
        continue
    if step[0] == 'final':   # a long copy-heavy answer at the session's depth (lookup drafts + DSpark at ~38K)
        if a.no_final:
            continue
        msgs = msgs + [{'role': 'assistant', 'content': 'I have read enough of these files.'},
                       {'role': 'user', 'content': 'Write out the code from the file part you read three tool results ago '
                                                   'again in full, unchanged except for a one-line comment above every '
                                                   'function. Output only the code.'}]
        rows.append(rec('final', 'turn', post(msgs, max(1, int(step[1] * a.dec_scale)))))
        continue
    n, mx = step
    if turn > 0:
        call = '{"tool": "read_file", "args": {"path": "...", "start": 1}}'
        reply = call if n < 200 else REPLIES[(turn - 1) % len(REPLIES)] + '\n' + call
        head = 'Tool result:\n'
        tail = ('\nContinue the review: say what this code does, name any problem you see in it, and show the corrected '
                'code.')
        want = n - ntok(reply) - ntok(head + tail) - 40   # ~40: the chat template's role tokens + the file headers
        body = code.take(want) if want > 24 else '(no output)'
        msgs = msgs + [{'role': 'assistant', 'content': reply}, {'role': 'user', 'content': head + body + tail}]
    rows.append(rec('t%02d' % turn, 'turn', post(msgs, max(1, int(mx * a.dec_scale)))))
    turn += 1
    if turn >= a.turns:
        break
tp = sum(r.get('eng_prompt_s', 0) for r in rows)
td = sum(r.get('dec_s', 0) for r in rows)
tw = sum(r.get('wall_s', 0) for r in rows)
nd = sum(r.get('dec_n', 0) for r in rows)
npass = sum(r.get('dsp_passes', 0) for r in rows)
cpu = sum(r.get('tg_cpu', 0) for r in rows)
print('TOTAL %s: wall %.1f s = prompts %.1f s + decode %.1f s (%d tokens, %.1f tok/s, %.2f tok/pass, %.1f ms/pass, '
      'cpu %.1f/pass)' % (a.label, tw, tp, td, nd, nd / max(td, 1e-9), nd / max(npass, 1), 1e3 * td / max(npass, 1),
                          cpu / max(npass, 1)), flush=True)
