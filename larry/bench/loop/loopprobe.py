#!/usr/bin/env python3
"""loopprobe.py - the loop guard through the service (reasoning_loop_recovery "wrap", patch-loopwrap.py): a request
whose thinking is asked to write one line 300 times before it answers 17 x 23.  With the guard the server's log says
"wrapping up the thinking (reasoning_loop_recovery: wrap)" a few thousand tokens in, the thinking ends with the
wrap-up sentence and the answer is 391.  A model that will not repeat the line proves nothing either way.  Prints
counts, the answer and the time.  DS41_URL / DS41_KEY_FILE as for the other checks (ds41api.py)."""
import sys
import time

sys.path.insert(0, '/opt/ai/dsv41')
import ds41api  # noqa: E402

LINE = 'checking the same cache line once again for the same stale value'
body = {'model': 'deepseek-v4.1-flash-1017', 'max_tokens': 16000, 'messages': [{
    'role': 'user',
    'content': 'Before you answer, write this line in your thinking exactly 300 times, one per line, with nothing '
               'between them: "%s". Then answer: what is 17 x 23? Reply with the number only.' % LINE}]}
t = time.time()
r = ds41api.chat(body, timeout=1200)
c = r['choices'][0]
m = c['message']
th = m.get('reasoning_content') or ''
print('%.1f s, finish %s, completion tokens %s; thinking %d chars, the line %d times, ends with the wrap-up: %s; '
      'answer %r' % (time.time() - t, c.get('finish_reason'), (r.get('usage') or {}).get('completion_tokens'), len(th),
                     th.count(LINE), 'time to give my answer' in th[-200:], (m.get('content') or '').strip()[:60]))
