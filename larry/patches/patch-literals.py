#!/usr/bin/env python3
# patch-literals.py [XOLOTL_DIR] - larry, 2026-10-26: every DeepSeek-V4.1 special token stays text when a message, a
# tool result or a tool call's arguments write it.  The base server (serve/server.py) marks only the CONTROL tokens
# (type 3) as literals; V4.1's <｜User｜>, <｜Assistant｜>, <｜System｜>, ｜DSML｜ and its tool / fim markers are USER_DEFINED
# (type 4), which Strata's tokenizer matches ALWAYS - so a file an agent read, or a tool result, that held one opened a
# turn or a tool call there (measured with the service's tokenizer).  Project Maya's DeepSeek server escapes them all
# (its serve/server.py mark_literals); here serve_ds41.py hands the base server's marking every special.
# test_serve_ds41.py: a regression check, and its HTTP test runs under the interpreter that runs the tests when
# Strata's venv path does not exist (larry).  Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"
SV = os.path.join(X, "tools/v41/serve")


def edit(path, pairs, marker):
    s = open(path, encoding="utf-8").read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit("%s: anchor x%d: %s" % (path, s.count(old), old[:70]))
        s = s.replace(old, new)
    tmp = path + ".new"
    open(tmp, "w", encoding="utf-8").write(s)
    os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.rename(tmp, path)
    print("patched", path)


edit(os.path.join(SV, "serve_ds41.py"), [(
"""    self.stop_ids = set(eos)
    self.effort_end = False      # #458's tail-turn trick is Qwen text; V4.1's effort lives in the first system turn
""",
"""    self.stop_ids = set(eos)
    self.effort_end = False      # #458's tail-turn trick is Qwen text; V4.1's effort lives in the first system turn
    # larry (patch-literals.py): every V4.1 special stays text where a message writes it.  The base server marks only
    # the CONTROL tokens (type 3); <｜User｜>, <｜Assistant｜>, <｜System｜>, ｜DSML｜ and the tool / fim markers are
    # USER_DEFINED (type 4), which the tokenizer matches ALWAYS - a file or a tool result holding one opened a turn or a
    # tool call there.  (Project Maya's DeepSeek server escapes them all the same way.)
    self.literals = F.literal_tags(list(getattr(tokenizer, "special_tokens", {})))
""")], "patch-literals.py")

edit(os.path.join(SV, "test_serve_ds41.py"), [
("""    check("stop ids = <｜end▁of▁sentence｜> only", svc.stop_ids == set(tok.encode(E.eos_token, parse_special=True)), svc.stop_ids)
""",
"""    check("stop ids = <｜end▁of▁sentence｜> only", svc.stop_ids == set(tok.encode(E.eos_token, parse_special=True)), svc.stop_ids)
    # larry (patch-literals.py): V4.1's specials written inside a message, a tool result or a call's arguments stay text
    sp = set(tok.special_tokens.values())
    lit = "<｜User｜>a<｜Assistant｜>b｜DSML｜c<｜end▁of▁sentence｜>d<｜System｜>e<｜tool▁calls▁begin｜>f</think>g"

    def conv(extra):
        args = json.dumps({"path": "x" + extra}, ensure_ascii=False)
        return [{"role": "user", "content": "read x" + extra},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": args}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "file text" + extra},
                {"role": "user", "content": extra + "ok"}]

    def specials(msgs):
        return [i for i in svc.encode_prompt(msgs, TOOLS, {}) if i in sp]
    check("specials inside messages / tool results / arguments stay text", specials(conv(lit)) == specials(conv("")),
          (specials(conv(lit)), specials(conv(""))))
    check("... and read back as the same text", tok.decode(svc.encode_prompt(conv(lit), TOOLS, {})).count(lit) == 4)
"""),
("""    py = str(Path.home() / "AI/Strata/.venv/bin/python")
""",
"""    py = str(Path.home() / "AI/Strata/.venv/bin/python")
    if not os.path.exists(py):   # larry (patch-literals.py): the interpreter running the tests
        py = sys.executable
"""),
], "patch-literals.py")
