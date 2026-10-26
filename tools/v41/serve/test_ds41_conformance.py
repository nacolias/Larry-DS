#!/usr/bin/env python3
"""test_ds41_conformance.py - larry, 2026-10-26: the DeepSeek-V4.1 server (serve_ds41.py on Xolotl's server) on the edge
cases Project Maya's serve/test_ds41_conformance.py covers (github.com/mw00/project-maya, MIT; the cases, not its code):
late system / developer turns, parallel tool results out of order (OpenAI and Anthropic), DSML string flags, DSML tags
inside parameter values.  Two differences from DeepSeek's reference are reported as notes, not failures: newlines
right after </think> are dropped (the base server's Qwen habit), and a long string argument arrives whole (no
argument streaming; the server's keep-alives hold idle clients).  CPU only, seconds:
  venv/bin/python tools/v41/serve/test_ds41_conformance.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_ds41 as D  # noqa: E402  (patches the server's modules)

E, F, S = D.E, D.F, D.S
T = D.Ds41Template()
FAILS = []


def check(name, ok, detail="", note=False):
    print(("ok   " if ok else "NOTE " if note else "FAIL ") + name + ("" if ok else "  " + str(detail)[:400]))
    if not ok and not note:
        FAILS.append(name)


TOOL = {"type": "function", "function": {"name": "lookup", "description": "Look something up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
TOOLS_FLAT = [TOOL["function"]]
BASE = [{"role": "system", "content": "System"}, {"role": "user", "content": "User"}]


def call(cid, q):
    return {"id": cid, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": q})}}


def render(msgs, tools=None, **kw):
    try:
        return T.render(msgs, tools, **kw), None
    except Exception as e:   # noqa: BLE001 - a probe: report it
        return None, repr(e)


# 1. parallel tool results out of order: results follow the calls' order (the encoder's sort), reasoning kept
r, err = render(BASE + [{"role": "assistant", "content": "", "reasoning_content": "Keep this reasoning",
                         "tool_calls": [call("a", "alpha"), call("b", "beta")]},
                        {"role": "tool", "tool_call_id": "b", "content": "RESULT_B"},
                        {"role": "tool", "tool_call_id": "a", "content": "RESULT_A"}], TOOLS_FLAT)
check("OpenAI parallel results in call order", r is not None and r.index("RESULT_A") < r.index("RESULT_B"), err or r[-300:])
check("the calling turn's reasoning kept", r is not None and "Keep this reasoning" + E.thinking_end_token in r, err)

# 2. a late system / developer turn keeps the system token
for role in ("system", "developer"):
    r, err = render(BASE + [{"role": "assistant", "content": "Old answer", "reasoning_content": "Old reasoning"},
                            {"role": role, "content": "New rules"}])
    check(f"late {role} turn renders as a system turn", r is not None and "<｜System｜>New rules" in r and
          "<｜User｜>New rules" not in r, err or r[-200:])

# 3. Anthropic: parallel tool_use, results out of order with text between them
req = {"model": "x", "max_tokens": 10, "system": "System", "messages": [
    {"role": "user", "content": "User"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "a", "name": "lookup", "input": {"q": "alpha"}},
        {"type": "tool_use", "id": "b", "name": "lookup", "input": {"q": "beta"}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "b", "content": "RESULT_B"},
        {"type": "text", "text": "between results"},
        {"type": "tool_result", "tool_use_id": "a", "content": "RESULT_A"}]}],
    "tools": [{"name": "lookup", "description": "Look something up", "input_schema": TOOL["function"]["parameters"]}]}
try:
    msgs, tools, kw = F.anthropic_to_messages(req)
    r, err = render(msgs, tools)
except Exception as e:   # noqa: BLE001
    r, err = None, repr(e)
ok = r is not None and "RESULT_A" in r and "RESULT_B" in r and "between results" in r
check("Anthropic parallel results render, text kept", ok, err or (r or "")[-400:])
if ok:
    check("Anthropic parallel results in call order", r.index("RESULT_A") < r.index("RESULT_B"), r[-400:])


# 4. the output parser (streaming, every split)
def run_parser(text, thinking, split, tools=TOOLS_FLAT):
    p = F.OutputParser(thinking=thinking, tools=tools)
    ev = []
    for i in range(0, len(text), split):
        ev += p.feed(text[i:i + split])
    ev += p.finish()
    content = "".join(e.text for e in ev if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    return content, calls, ev


P_OPEN, P_CLOSE = D.P_OPEN, D.P_CLOSE


def dsml_param(name, value, flag):
    return (f"\n\n{D.CALLS_OPEN}\n{D.INVOKE_OPEN} name=\"lookup\">\n{P_OPEN} name=\"{name}\" string=\"{flag}\">{value}"
            f"{P_CLOSE}\n{D.INVOKE_CLOSE}\n{D.CALLS_CLOSE}")


for value, flag, expected in [("123", "true", "123"), ("true", "true", "true"), ("42", "false", 42),
                              ('{"nested": [true, null, 2]}', "false", {"nested": [True, None, 2]}),
                              ("\nhello\n", "true", "\nhello\n")]:
    bad = [s for s in (1, 7, 1000) if run_parser(dsml_param("q", value, flag), False, s)[1] != [("lookup", {"q": expected})]]
    check(f"DSML string={flag} {value!r} -> {expected!r}", not bad, run_parser(dsml_param("q", value, flag), False, 7)[1])

# 5. DSML tags inside a parameter value, as DeepSeek's encoder writes them, parse as DeepSeek's parser reads them
for tag in (D.CALLS_OPEN, D.CALLS_CLOSE, D.INVOKE_OPEN + ' name="nested">', D.INVOKE_CLOSE,
            P_OPEN + ' name="nested" string="true">', "<｜DSML｜ unknown>"):
    value = f"literal {tag} text"
    params = E.encode_arguments_to_dsml({"arguments": {"q": value}})
    text = f"\n\n{D.CALLS_OPEN}\n{D.INVOKE_OPEN} name=\"lookup\">\n{params}\n{D.INVOKE_CLOSE}\n{D.CALLS_CLOSE}"
    ref = E.parse_message_from_completion_text(text + E.eos_token, "chat")
    want = json.loads(ref["tool_calls"][0]["function"]["arguments"]) if ref.get("tool_calls") else None
    got = [run_parser(text, False, s)[1] for s in (1, 7, 1000)]
    check(f"a DSML tag inside a value parses as DeepSeek's parser does ({tag[:24]!r})",
          all(g == [("lookup", want)] for g in got), (want, got[1]))

# 6. newlines after </think> stay in the answer
for thinking in (False, True):
    text = ("reason" + E.thinking_end_token if thinking else "") + "\n\nanswer"
    outs = {run_parser(text, thinking, s, tools=None)[0] for s in (1, 7, 100)}
    check(f"leading newlines kept (thinking={thinking})", outs == {"\n\nanswer"}, outs, note=thinking)

# 7. a long string argument streams before its closing tag (tool_args events before the call completes)
body = "x" * 4000
text = dsml_param("q", body, "true")
_, calls, ev = run_parser(text, False, 64)
kinds = [e.kind for e in ev]
first_args = kinds.index("tool_args") if "tool_args" in kinds else -1
check("long string argument parses", calls == [("lookup", {"q": body})], calls[:1])
check("long string argument streams before the call ends", 0 <= first_args < len(kinds) - 2 and kinds.count("tool_args") > 4,
      (first_args, kinds.count("tool_args"), len(kinds)), note=True)

print(f"{'FAILED ' + str(len(FAILS)) + ': ' + ', '.join(FAILS) if FAILS else 'all passed'}")
sys.exit(1 if FAILS else 0)
