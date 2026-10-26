#!/usr/bin/env python3
"""gpqa.py - GPQA Diamond through the ds41 server the way DeepSeek's own eval runs it (DeepInfer eval/gpqa + run_eval.py):
its prompt ("Answer the following multiple choice question. The last line of your response should be of the following
format: 'ANSWER: $LETTER' ... Think step by step before answering." + the question with its four choices) as one user
turn, thinking at reasoning effort 100, temperature 1.0, top-p 1.0, 144K tokens and no thinking budget, scored by
DeepSeek's rule judge (the last "ANSWER: X" line, else the last \\boxed{X}, else the last standalone A-D after
</think>).  One sample a question instead of DeepSeek's 8.  Same client as aime.py.

Resumable: each finished problem is one line of OUT; task ids already there are skipped.  It stops before the next
problem when the file STOP exists next to OUT, or once --deadline (UTC, HH:MM) has passed.  It YIELDS to anyone else:
the server runs one request at a time, so when its /metrics shows another request queued, the client closes its own
(the server cancels it) and stops - that problem runs again on the next start, as after a kill.
  gpqa.py [--port 18080] [--out FILE] [--deadline HH:MM] [--limit N]     run
  gpqa.py --score [--out FILE]                                            print the score of what OUT holds
Run with /opt/ai/dsv41/venv/bin/python -I.  Data: data/gpqa_diamond.jsonl (hendrydong/gpqa_diamond_mc, the public
mirror DeepInfer names, normalised as its loader does: the mirror's trailing answer-format line stripped, \\boxed{X} ->
X, task ids gpqa_diamond_<row>).
"""
import argparse
import datetime
import http.client
import json
import os
import re
import socket
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='172.23.0.1')
ap.add_argument('--port', type=int, default=18080)
ap.add_argument('--data', default='/opt/ai/dsv41/bench/quality/data/gpqa_diamond.jsonl')
ap.add_argument('--out', default='/opt/ai/dsv41/bench/quality/out/gpqa-1.jsonl')
ap.add_argument('--deadline', default=None, help='UTC HH:MM: start no problem after it')
ap.add_argument('--max-len', type=int, default=144 * 1024, help='prompt + output tokens (DeepInfer --max-len)')
ap.add_argument('--effort', default='max')
ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--score', action='store_true')
a = ap.parse_args()

INSTR = ("Answer the following multiple choice question. The last line of your response should be "
         "of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
         "Think step by step before answering.\n\n{question}")


# DeepSeek's rule judge (DeepInfer eval/gpqa/bench.py), unchanged
_ANSWER_LINE_RE = re.compile(r"(?im)^\s*ANSWER\s*[:\uff1a]\s*\(?([A-D])\b")
_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def extract_answer_with_source(text):
    if not text:
        return None, 'none'
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return matches[-1].upper(), 'answer_line'
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].upper(), 'boxed'
    tail = text.rsplit('</think>', 1)[-1] if '</think>' in text else text[-400:]
    matches = _STANDALONE_RE.findall(tail)
    if matches:
        return matches[-1].upper(), 'tail_letter'
    return None, 'none'


def is_correct(pred, gold):
    return pred is not None and pred.upper() == str(gold).strip().upper()


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done[r['task_id']] = r
    return done


def score(path, problems):
    done = load_done(path)
    rows = [done[p['task_id']] for p in problems if p['task_id'] in done]
    if not rows:
        print('nothing scored yet')
        return
    ok = sum(r['correct'] for r in rows)
    toks = [r['completion_tokens'] for r in rows]
    secs = sum(r['wall_s'] for r in rows)
    src = {}
    for r in rows:
        src[r['extract_source']] = src.get(r['extract_source'], 0) + 1
    fin = {}
    for r in rows:
        fin[r['finish']] = fin.get(r['finish'], 0) + 1
    print('GPQA Diamond, %d of %d questions done: %d correct = %.1f %% (Pass@1, one sample a question)' % (
        len(rows), len(problems), ok, 100.0 * ok / len(rows)))
    print('  answers from: %s; finish: %s' % (src, fin))
    print('  output tokens: mean %.0f, median %.0f, max %d; %.1f h of generation, mean %.1f tok/s' % (
        sum(toks) / len(toks), sorted(toks)[len(toks) // 2], max(toks), secs / 3600,
        sum(toks) / max(secs, 1e-9)))
    for r in rows:
        print('  %-16s gold %4s pred %-6s %-5s %6d tokens %6.0f s %s' % (
            r['task_id'], r['gold'], r['pred'], 'OK' if r['correct'] else 'WRONG', r['completion_tokens'],
            r['wall_s'], r['finish']))


def api_key():
    for line in open('/opt/docker/vllm/.env'):
        if line.startswith('VLLM_API_KEY'):
            return line.split('=', 1)[1].strip()
    raise SystemExit('no VLLM_API_KEY in /opt/docker/vllm/.env')


def prompt_tokens(question):
    sys.path.insert(0, '/opt/ai/dsv41')
    import tok
    global _tk
    if '_tk' not in globals():
        _tk = tok.tokenizer()
    text = tok.E.encode_messages([{'role': 'user', 'content': question}], thinking_mode='thinking',
                                 reasoning_effort=100)
    return len(_tk.encode(text, parse_special=True))


class Yielded(Exception):
    pass


def queued(key):
    """Requests waiting behind ours (the server's /metrics), 0 when it cannot tell."""
    try:
        c = http.client.HTTPConnection(a.host, a.port, timeout=10)
        c.request('GET', '/metrics', headers={'Authorization': 'Bearer ' + key})
        r = c.getresponse()
        d = json.loads(r.read()) if r.status == 200 else {}
        c.close()
        return int((d.get('live') or {}).get('queued') or 0)
    except Exception:
        return 0


def run_one(p, key):
    q = INSTR.format(question=p['question'].strip())
    n_prompt = prompt_tokens(q)
    body = {'model': 'deepseek-v4.1-flash-1017', 'messages': [{'role': 'user', 'content': q}], 'stream': True,
            'stream_options': {'include_usage': True}, 'temperature': 1.0, 'top_p': 1.0,
            'reasoning_effort': a.effort, 'reasoning_budget_tokens': 0, 'max_tokens': a.max_len - n_prompt}
    conn = http.client.HTTPConnection(a.host, a.port, timeout=1800)
    t0 = time.time()
    conn.request('POST', '/v1/chat/completions', json.dumps(body),
                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    sock = conn.sock   # getresponse() hands the socket to the response (conn.sock = None for a streamed reply)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError('HTTP %d: %s' % (resp.status, resp.read()[:500]))
    done_ev, yielded = threading.Event(), []

    def watch():   # another request queued behind ours: close ours (the server cancels it at the closed socket)
        while not done_ev.wait(10):
            if queued(key) > 0:
                yielded.append(time.time())
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                return
    threading.Thread(target=watch, daemon=True).start()
    reasoning, content, finish, usage, timings, first = [], [], None, None, None, None
    got_done = False
    last_note = t0
    while True:
        try:
            line = resp.readline()
        except OSError:
            line = b''
        if not line:
            break
        line = line.strip()
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            got_done = True
            break
        ev = json.loads(data)
        if ev.get('usage'):
            usage = ev['usage']
        if ev.get('timings'):
            timings = ev['timings']
        for ch in ev.get('choices') or []:
            d = ch.get('delta') or {}
            if d.get('reasoning_content'):
                reasoning.append(d['reasoning_content'])
            if d.get('content'):
                content.append(d['content'])
            if (d.get('reasoning_content') or d.get('content')) and first is None:
                first = time.time() - t0
            if ch.get('finish_reason'):
                finish = ch['finish_reason']
        if time.time() - last_note > 300:
            last_note = time.time()
            print('    ... %s: %.0f s, ~%d chunks' % (p['task_id'], last_note - t0, len(reasoning) + len(content)),
                  flush=True)
    done_ev.set()
    conn.close()
    if yielded and not got_done:
        raise Yielded()
    wall = time.time() - t0
    raw = ''.join(reasoning) + '</think>' + ''.join(content)
    pred, source = extract_answer_with_source(raw)
    return {'task_id': p['task_id'], 'gold': p['answer'], 'pred': pred, 'extract_source': source,
            'correct': is_correct(pred, p['answer']), 'finish': finish, 'prompt_tokens': n_prompt,
            'completion_tokens': (usage or {}).get('completion_tokens', 0), 'usage': usage, 'timings': timings,
            'wall_s': round(wall, 1), 'ttft_s': round(first or 0, 2), 'effort': a.effort,
            'started': datetime.datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S'),
            'reasoning': ''.join(reasoning), 'content': ''.join(content), 'yield': bool(yielded)}


problems = [json.loads(line) for line in open(a.data) if line.strip()]
if a.limit:
    problems = problems[:a.limit]
if a.score:
    score(a.out, problems)
    sys.exit(0)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
stop = os.path.join(os.path.dirname(a.out), 'STOP')
key = api_key()
done = load_done(a.out)
todo = [p for p in problems if p['task_id'] not in done]
print('%s: %d problems, %d done, %d to go -> %s' % (time.strftime('%H:%M:%S'), len(problems), len(done), len(todo),
                                                    a.out), flush=True)
for p in todo:
    if os.path.exists(stop):
        print('STOP file - stopping', flush=True)
        break
    if a.deadline:
        now = datetime.datetime.utcnow()
        hh, mm = (int(v) for v in a.deadline.split(':'))
        if (now.hour, now.minute) >= (hh, mm):
            print('deadline %s UTC passed - stopping' % a.deadline, flush=True)
            break
    print('%s %s ...' % (time.strftime('%H:%M:%S'), p['task_id']), flush=True)
    try:
        r = run_one(p, key)
    except Yielded:
        print('  %s: another request is waiting - cancelled ours, stopping (run again to resume)' % p['task_id'], flush=True)
        break
    except Exception as e:  # the server went away (a restart): leave the problem for the next start
        print('  %s failed: %r - stopping' % (p['task_id'], e), flush=True)
        break
    with open(a.out, 'a') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('  %s: %s (gold %s, pred %s via %s), %d tokens in %.0f s, finish %s' % (
        p['task_id'], 'OK' if r['correct'] else 'WRONG', r['gold'], r['pred'], r['extract_source'],
        r['completion_tokens'], r['wall_s'], r['finish']), flush=True)
    if r['yield']:
        print('another request is waiting - stopping (run again to resume)', flush=True)
        break
score(a.out, problems)
