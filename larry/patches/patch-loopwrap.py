#!/usr/bin/env python3
# patch-loopwrap.py [XOLOTL_DIR] - larry, 2026-10-29: "reasoning_loop_recovery": "wrap", a third mode for the server's
# loop detector (#728 / #1753): when the thinking repeats whole passages, it is closed the way the thinking budget closes
# it (REASONING_WRAP_UP and </think>, the same prompt and sampler: the engine keeps the prefix it holds) and the model
# answers.  The other two modes do not suit an agent at depth: "recover" reads the whole prompt again with the low-effort
# system sentence (~30 s at a 100K-token context, and the next turn reads the original again), "stop" ends the reply with
# no answer.  The agent session of 2026-10-29 05:39-05:59 had nine loops of ~24,600 tokens (~2 min each, 99.9 % of the
# lookup drafts accepted) that ran into the 24,576-token budget, and the loop text stayed in the conversation, so the
# next turns looped again.  Also adds the mode's tests to serve/test_reasoning_loop_recovery.py.  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
S = os.path.join(X, "serve/server.py")
T = os.path.join(X, "serve/test_reasoning_loop_recovery.py")
s, t = open(S).read(), open(T).read()
if "patch-loopwrap.py" in s:
    print("already patched")
    sys.exit(0)


def rep(s, old, new, what):
    if s.count(old) != 1:
        sys.exit("%s: anchor x%d: %s" % (what, s.count(old), old[:90]))
    return s.replace(old, new)


s = rep(s, """# low-effort instruction in place of the xhigh one.  Both off by default.
""", """# low-effort instruction in place of the xhigh one.  Both off by default.
# larry (patch-loopwrap.py): "wrap" closes the thinking there the way the thinking budget does, and the model answers.
""", "doc")
s = rep(s, """        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
""", """        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
        loop_wrap = False                               # larry (patch-loopwrap.py): a loop closed the thinking
""", "init")
s = rep(s, """                                        if self.reasoning_loop_recovery == "stop":
                                            looped = True        # #728: end the reply here, as for a repeated token
                                            break
""", """                                        if self.reasoning_loop_recovery == "stop":
                                            looped = True        # #728: end the reply here, as for a repeated token
                                            break
                                        if self.reasoning_loop_recovery == "wrap":
                                            # larry (patch-loopwrap.py): close the thinking here as the budget does
                                            wrap = loop_wrap = True
                                            break
""", "detect")
s = rep(s, """                        if wrap:
                            print(f"[strata] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
""", """                        if wrap and loop_wrap:
                            # larry (patch-loopwrap.py): the loop detector, not the budget, closed the thinking
                            recovery_count += 1
                            with self.status_lock:
                                st["reasoning_recoveries"] = recovery_count
                            if trace is not None:
                                trace["reasoning_recoveries"] = recovery_count
                                trace["reasoning_repeat_coverage"] = round(repeat_coverage, 3)
                            print(f"[strata] the thinking repeated the same passages (coverage={repeat_coverage:.3f}) "
                                  f"at {n} tokens: wrapping up the thinking (reasoning_loop_recovery: wrap)", flush=True)
                        elif wrap:
                            print(f"[strata] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
""", "print")
s = rep(s, """    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover"):
        raise SystemExit("[strata] config \\"reasoning_loop_recovery\\" must be false, \\"stop\\" or \\"recover\\", "
                         f"not {recovery!r}")
""", """    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:                                    # | "wrap" (larry, patch-loopwrap.py)
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover", "wrap"):
        raise SystemExit("[strata] config \\"reasoning_loop_recovery\\" must be false, \\"stop\\", \\"recover\\" or "
                         f"\\"wrap\\", not {recovery!r}")
""", "config")

t = rep(t, """

if __name__ == '__main__':
""", """
    # larry (patch-loopwrap.py): "wrap" closes a looping thinking the way the thinking budget does
    def run_wrap(self, scripts, sampling=None, batch=0):
        engine = RecordingEngine(self.tok, scripts)
        engine.batch = batch
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = "wrap"
        events = list(service.run(self.ids, True, [], 40000, sampling or {}, threading.Event()))
        return engine, service, events

    def test_wrap_mode_closes_the_thinking_and_the_model_answers(self):
        engine, service, events = self.run_wrap([self.loop, 'Complete solution.'], {'temperature': .6, 'top_p': .95})
        self.assertEqual(len(engine.calls), 2)
        first, second = engine.calls[0][0], engine.calls[1][0]
        self.assertEqual(second[:len(first)], first)             # the same prompt: the engine keeps its prefix
        added = self.tok.decode(second[len(first):])
        self.assertTrue(added.startswith(self.loop[:200]))       # what was generated, then the wrap-up
        self.assertTrue(added.endswith('</think>\\n\\n'))
        self.assertLess(len(second) - len(first), 20000)         # long before the budget
        self.assertEqual(engine.sampling_calls[1], engine.sampling_calls[0])   # the sampler unchanged
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        self.assertEqual(content, 'Complete solution.')
        self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(service.history[-1]['reasoning_recoveries'], 1)
        self.assertGreaterEqual(service.history[-1]['reasoning_repeat_coverage'], .25)

    def test_wrap_mode_closes_a_loop_written_as_one_long_word(self):
        engine, _, events = self.run_wrap(['9^104 = ' + '58747693387' * 3000, 'Answer.'])
        self.assertEqual(len(engine.calls), 2)
        self.assertLess(len(engine.calls[1][0]) - len(engine.calls[0][0]), 5000)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)

    def test_wrap_mode_leaves_novel_reasoning_alone(self):
        work = ' '.join(str(i) for i in range(6000))
        engine, _, events = self.run_wrap([work + '\\n</think>\\nAnswer.'])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_wrap_mode_with_batch_slots(self):
        engine, service, events = self.run_wrap([self.loop, 'Answer.'], batch=2)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertFalse(service.status['busy'])
        self.assertEqual(service.live_reqs, {})


if __name__ == '__main__':
""", "tests")
open(S, "w").write(s)
open(T, "w").write(t)
print("patched", S, T)
