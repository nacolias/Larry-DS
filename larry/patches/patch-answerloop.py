#!/usr/bin/env python3
# patch-answerloop.py [XOLOTL_DIR] - larry, 2026-10-29: "answer_loop_stop": <coverage> - an answer that repeats whole
# passages is ended there ("length").  The loop detector (#728 / #1753; patch-loopwrap.py's "wrap") reads the thinking
# only: a probe through the service once wrote its 300 repeated lines in the answer and ran to max_tokens (16,000 tokens,
# 82 s).  The same measure (the share of the last 2,000 words inside 12-word passages seen three times, or a 2,048-
# character tail with a period of <= 64), every 512 tokens of an answer, against a much higher bar than the thinking's
# 0.25 (the config's value; 0.9 in larry-ds41.json): code and tables that repeat lines vary their names and numbers, a
# loop does not.  Answers under 2,000 words are never measured.  Also adds its tests to
# serve/test_reasoning_loop_recovery.py.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
S = os.path.join(X, "serve/server.py")
T = os.path.join(X, "serve/test_reasoning_loop_recovery.py")
s, t = open(S).read(), open(T).read()
if "patch-answerloop.py" in s:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    return s.replace(old, new)


s = rep(s, """        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply (0: off)
""", """        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply (0: off)
        self.answer_loop_stop = 0.0                      # larry (patch-answerloop.py): an answer's loop coverage that ends it
""", "init")
s = rep(s, """        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
""", """        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
        answer_text, answer_looped, answer_cov, next_answer_check = "", False, 0.0, 0   # larry (patch-answerloop.py)
""", "request vars")
s = rep(s, """                                    if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                        reasoning_text += ev.text or ""
""", """                                    if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                        reasoning_text += ev.text or ""
                                    if self.answer_loop_stop and ev.kind == "content":   # (patch-answerloop.py)
                                        answer_text += ev.text or ""
""", "accumulate")
s = rep(s, """                                if budget and parser.state == "reasoning":
                                    thought += 1""", """                                # larry (patch-answerloop.py): an answer that repeats whole passages ends here ("length")
                                if (self.answer_loop_stop and parser.state == "content" and n >= next_answer_check
                                        and not parser.buf and not parser.pending and not detok.pending()):
                                    next_answer_check = n + LOOP_CHECK_EVERY
                                    answer_cov = reasoning_repeat_coverage(answer_text)
                                    if answer_cov >= self.answer_loop_stop:
                                        answer_looped = True
                                        break
                                if budget and parser.state == "reasoning":
                                    thought += 1""", "check")
s = rep(s, """                    elif repeated:
                        print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) \"""",
        """                    elif answer_looped:   # larry (patch-answerloop.py)
                        print(f"[strata] the answer repeated the same passages (coverage={answer_cov:.3f}) at {n} "
                              f"tokens: ended as \\"length\\" (answer_loop_stop {self.answer_loop_stop} in the config; 0 "
                              "turns this off)", flush=True)
                    elif repeated:
                        print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) \"""", "print")
s = rep(s, """    svc.repeat_stop_tokens = rs
""", """    svc.repeat_stop_tokens = rs
    als = cfg.get("answer_loop_stop", 0)   # larry (patch-answerloop.py): 0 (off) or a coverage in (0, 1]
    if isinstance(als, bool) or not isinstance(als, (int, float)) or not 0 <= als <= 1:
        raise SystemExit(f"[strata] config \\"answer_loop_stop\\" must be a number from 0 (off) to 1, not {als!r}")
    svc.answer_loop_stop = float(als)
""", "config")

t = rep(t, """

if __name__ == '__main__':
""", """
    # larry (patch-answerloop.py): "answer_loop_stop" ends an answer that repeats whole passages
    def run_answer(self, script, threshold=0.9, max_new=40000):
        engine = RecordingEngine(self.tok, [script])
        service = Service(engine, self.tok, None)
        service.answer_loop_stop = threshold
        ids = self.tok.encode('<|im_start|>user\\nTask<|im_end|>\\n<|im_start|>assistant\\n', parse_special=True)
        events = list(service.run(ids, False, [], max_new, {}, threading.Event()))
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        return engine, events, content

    def test_answer_loop_is_ended(self):
        engine, events, content = self.run_answer(self.loop)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 20000)
        self.assertTrue(content.startswith(self.loop[:200]))

    def test_answer_loop_as_one_long_word_is_ended(self):
        _, events, _ = self.run_answer('9^104 = ' + '58747693387' * 3000)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 5000)

    def test_repetitive_code_is_not_ended(self):
        code = ''.join('    assert square(%d) == %d, "square of %d"\\n' % (i, i * i, i) for i in range(600))
        _, events, content = self.run_answer(code)
        self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(content, code)

    def test_novel_answer_is_not_ended(self):
        work = ' '.join(str(i) for i in range(6000))
        _, events, content = self.run_answer(work)
        self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(content, work)

    def test_answer_guard_off_by_default(self):
        _, events, _ = self.run_answer(self.loop, threshold=0.0, max_new=16384)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)


if __name__ == '__main__':
""", "tests")
open(S, "w").write(s)
open(T, "w").write(t)
print("patched", S, T)
