#!/usr/bin/env python3
"""Tests for serve_ds41.py (CPU, seconds).  Run under Strata's venv:
  /home/mal/AI/Strata/.venv/bin/python tools/v41/serve/test_serve_ds41.py [TOKENIZER_DIR]
TOKENIZER_DIR (export_tokenizer.py's output) enables the tokenizer and the HTTP tests (server --engine mock)."""
import itertools
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_ds41 as D  # noqa: E402  (patches the server's modules)

E, F = D.E, D.F
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:300]))
    if not ok:
        FAILS.append(name)


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
                         "required": ["path"]}}}]


def t_template():
    T = D.Ds41Template()
    chats = [
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello", "reasoning_content": "greet"}, {"role": "user", "content": "again"}],
        [{"role": "user", "content": "read a.txt"},
         {"role": "assistant", "content": "", "reasoning_content": "use the tool",
          "tool_calls": [{"id": "c1", "type": "function",
                          "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt", "lines": 3})}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "AAA"}],
    ]
    for i, msgs in enumerate(chats):
        tools = TOOLS if i == 2 else None
        for mode, kw in (("thinking", {}), ("chat", {"enable_thinking": False})):
            ref_msgs = [dict(m) for m in msgs]
            if tools:
                ref_msgs.insert(0, {"role": "system", "content": "", "tools": tools})
            ref = E.encode_messages(ref_msgs, thinking_mode=mode)
            got = T.render(msgs, tools=tools, **kw)
            if tools:   # the server's flat tool format must render the same
                flat = T.render(msgs, tools=[t["function"] for t in tools], **kw)
                check(f"template chat{i} {mode} flat tools == OpenAI tools", flat == got)
            check(f"template chat{i} {mode} == encode_messages", got == ref, (got[-200:], ref[-200:]))
            hist = T.render(msgs, tools=tools, add_generation_prompt=False, **kw)
            check(f"template chat{i} {mode} no-gen is a prefix", got.startswith(hist) and len(hist) < len(got))
    # content parts, effort
    got = T.render([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}])
    check("template content parts joined", got == E.encode_messages([{"role": "user", "content": "xy"}], "thinking"))
    for eff, n in (("max", 100), (30, 30), ("low", 50), ("high", 75), ("xhigh", 75)):
        got = T.render([{"role": "user", "content": "q"}], reasoning_effort=eff)
        check(f"effort {eff!r} -> {n}", f"Reasoning Effort: {n} " in got, got[:120])
    got = T.render([{"role": "user", "content": "q"}])
    check("effort default -> 75", "Reasoning Effort: 75 " in got)


def t_effort_kwargs():
    for v, want in (("high", {"reasoning_effort": 75}), ("max", {"reasoning_effort": 100}), ("25", {"reasoning_effort": 25}),
                    ("none", {"enable_thinking": False}), ("0", {"enable_thinking": False}), (False, {"enable_thinking": False}),
                    ("medium", {"reasoning_effort": 62}), (None, {})):
        check(f"effort_kwargs({v!r})", D.effort_kwargs(v) == want, D.effort_kwargs(v))
    try:
        D.effort_kwargs("turbo"); check("effort_kwargs rejects unknown", False)
    except ValueError:
        check("effort_kwargs rejects unknown", True)


def dsml(calls):
    inv = []
    for name, args in calls:
        ps = []
        for k, v in args.items():
            s = isinstance(v, str)
            ps.append(f'<{E.dsml_token}{E.tool_parameter_tag_name} name="{k}" string="{"true" if s else "false"}">'
                      f'{v if s else json.dumps(v)}</{E.dsml_token}{E.tool_parameter_tag_name}>')
        inv.append(f'<{E.dsml_token}{E.tool_call_tag_name} name="{name}">\n' + "\n".join(ps) +
                   f'\n</{E.dsml_token}{E.tool_call_tag_name}>')
    return f"\n\n{D.CALLS_OPEN}\n" + "\n".join(inv) + f"\n{D.CALLS_CLOSE}"


def run_parser(text, thinking, split):
    p = F.OutputParser(thinking=thinking, tools=TOOLS)
    ev = []
    for i in range(0, len(text), split):
        ev += p.feed(text[i:i + split])
    ev += p.finish()
    content = "".join(e.text for e in ev if e.kind == "content")
    reasoning = "".join(e.text for e in ev if e.kind == "reasoning")
    calls = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    return reasoning, content, calls


def t_parser():
    calls = [("read_file", {"path": "a b.txt", "lines": 3}),
             ("read_file", {"path": "x</y>", "lines": [1, 2], "opts": {"k": True}})]
    body = "I'll look." + dsml(calls)
    for thinking in (True, False):
        text = ("checking files" + E.thinking_end_token if thinking else "") + body
        ref = E.parse_message_from_completion_text(text + E.eos_token, "thinking" if thinking else "chat")
        ref_calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]]
        bad = []
        for split in range(1, 40):
            r, c, k = run_parser(text, thinking, split)
            if k != ref_calls or c.strip() != ref["content"].strip() or (thinking and r != ref["reasoning_content"]):
                bad.append((split, r, c, k))
        check(f"DSML parse == DeepSeek's parser at every split 1..39 (thinking={thinking})", not bad, bad[:1] or ref)
    # plain answer, and text that only looks like a wrapper start
    r, c, k = run_parser("plain answer <｜DSML not a tag", False, 3)
    check("non-DSML text passes through", c == "plain answer <｜DSML not a tag" and not k, (c, k))
    esc = "a&lt;" + D.P_CLOSE[1:] + "b"
    check("DwarfStar escape undone", D.unescape(esc) == "a" + D.P_CLOSE + "b", D.unescape(esc))


def load_tok(tok_dir):
    import strata_tokenizer as ST
    tokens = json.load(open(tok_dir / "vocab.json"))
    tokens = [t for t, _ in sorted(tokens.items(), key=lambda kv: kv[1])] if isinstance(tokens, dict) else tokens
    merges = (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.load(open(tok_dir / "token_type.json"))
    return ST.Tokenizer(tokens, merges, types)   # = serve_ds41.Ds41Tokenizer (patched)


def t_tokenizer(tok_dir):
    tok = load_tok(tok_dir)
    p = D.Ds41Template().render([{"role": "user", "content": "hi"}])
    ids = tok.encode(p, parse_special=True)
    check("tokenizer: specials are single tokens", len(tok.encode(E.eos_token, parse_special=True)) == 1 and
          len(tok.encode(E.USER_SP_TOKEN, parse_special=True)) == 1, ids[:8])
    check("tokenizer round trip", tok.decode(ids) == p)
    import serve.server as S
    svc = S.Service(S.MockEngine(tok, "x"), tok, D.Ds41Template())
    check("stop ids = <｜end▁of▁sentence｜> only", svc.stop_ids == set(tok.encode(E.eos_token, parse_special=True)), svc.stop_ids)
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


def t_images(tok_dir):
    """larry (patch-vision-server.py): a message's pictures reach DeepSeek's encoder in place, and Service.prepare gives
    each its span of image tokens in the encoder's order (a stub encoder: no model needed)"""
    import serve.server as S
    tok = load_tok(tok_dir)
    img = tok.encode(E.IMAGE_PLACEHOLDER, parse_special=True)
    check("image placeholder is one token", len(img) == 1, img)
    img = img[0]
    pic = lambda src: {"type": "image", "source": src}
    txt = lambda t: {"type": "text", "text": t}
    check("content without pictures stays its text", D._content([txt("a"), txt("b")]) == "ab")
    got = D._content([txt("see "), pic("/p/a.png"), txt(" and "), pic("/p/b.png")])
    check("pictures kept in place for the encoder", got == [txt("see "), pic("/p/a.png"), txt(" and "), pic("/p/b.png")], got)
    # parallel tool results arrive out of call order (DeepSeek's encoder sorts them by call id); a typed marker is text
    calls = [{"id": c, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f})}}
             for c, f in (("c1", "a.png"), ("c2", "b.png"))]
    msgs = [{"role": "user", "content": [txt("compare "), pic("/p/u.png")]},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c2", "content": [txt("RESULT-B2"), pic("/p/b.png")]},
            {"role": "tool", "tool_call_id": "c1", "content": [txt("RESULT-A1"), pic("/p/a.png")]},
            {"role": "user", "content": "the " + E.IMAGE_PLACEHOLDER + " marker, typed"}]
    order = D.ds41_images_of(msgs)
    check("images_of in the encoder's order", order == ["/p/u.png", "/p/a.png", "/p/b.png"], order)

    class StubVision:   # (file, image tokens) per picture, as Vision.encode
        def __init__(self, d):
            self.dir, self.n = d, {"/p/u.png": 5, "/p/a.png": 7, "/p/b.png": 9}

        def encode(self, src):
            f = Path(self.dir) / (Path(src).stem * 16 + ".sve")
            f.write_bytes(b"x" * self.n[src])
            return str(f), self.n[src]
    with tempfile.TemporaryDirectory() as d:
        svc = S.Service(S.MockEngine(tok, "x"), tok, D.Ds41Template(), vision=StubVision(d))
        try:
            ids = svc.prepare(msgs, TOOLS, {})[0]
            keys = svc.embeddings.keys
        finally:
            svc.drop_embeddings()
        base = svc.encode_prompt(msgs, TOOLS, {})   # one placeholder per picture

    def show(ids):   # the prompt with each run of image tokens as [IMGn]
        return "".join(f"[IMG{len(g)}]" if k else tok.decode(g)
                       for k, g in ((k, list(g)) for k, g in itertools.groupby(ids, key=lambda t: t == img)))
    shown, want = show(ids), show(base)
    for n in (5, 7, 9):
        want = want.replace("[IMG1]", f"[IMG{n}]", 1)
    check("each picture its span in its placeholder's place", shown == want, (shown[-300:], want[-300:]))
    at = [shown.find(t) for t in ("compare", "[IMG5]", "RESULT-A1", "[IMG7]", "RESULT-B2", "[IMG9]")]
    check("... the tool results in call order, each with its picture", -1 not in at and at == sorted(at), at)
    check("... a typed marker stays text", "the " + E.IMAGE_PLACEHOLDER + " marker, typed" in shown, shown[-200:])
    check("... one picture key each", keys == ["u" * 16, "a" * 16, "b" * 16], keys)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def t_http(tok_dir):
    port = free_port()
    script = "need the file" + E.thinking_end_token + dsml([("read_file", {"path": "a.txt", "lines": 3})])
    py = str(Path.home() / "AI/Strata/.venv/bin/python")
    if not os.path.exists(py):   # larry (patch-literals.py): the interpreter running the tests
        py = sys.executable
    proc = subprocess.Popen([py, str(HERE / "serve_ds41.py"), "--engine", "mock", "--port", str(port),
                             "--tokenizer", str(tok_dir), "--script", script, "--script", script, "--script", "done" + E.thinking_end_token + "The file says AAA."],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        msgs = [{"role": "user", "content": "read a.txt"}]
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs, "tools": TOOLS})
        ch = r["choices"][0]
        tc = (ch["message"].get("tool_calls") or [{}])[0].get("function", {})
        check("HTTP OpenAI tool call", ch.get("finish_reason") == "tool_calls" and tc.get("name") == "read_file" and
              json.loads(tc.get("arguments", "{}")) == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/messages", {"model": "x", "max_tokens": 200, "messages": msgs,
                                         "tools": [{"name": "read_file", "description": "Read a file",
                                                    "input_schema": TOOLS[0]["function"]["parameters"]}]})
        tu = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
        check("HTTP Anthropic tool_use", tu and tu[0]["name"] == "read_file" and tu[0]["input"] == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs})
        check("HTTP plain answer after the scripted call", "AAA" in (r["choices"][0]["message"].get("content") or ""), r)
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill(); out = proc.communicate()[0]
        if FAILS:
            print("---- server output (tail) ----\n" + out[-3000:])


if __name__ == "__main__":
    t_template()
    t_effort_kwargs()
    t_parser()
    if len(sys.argv) > 1:
        t_tokenizer(Path(sys.argv[1]))
        t_images(Path(sys.argv[1]))
        t_http(Path(sys.argv[1]))
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
