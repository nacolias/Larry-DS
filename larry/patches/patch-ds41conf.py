#!/usr/bin/env python3
# patch-ds41conf.py [XOLOTL_DIR] - larry, 2026-10-26: the DeepSeek-V4.1 server on the edge cases Project Maya's
# serve/test_ds41_conformance.py covers, where ours failed (bench/serve/ds41conf.py found them):
#   1. a `developer` message (OpenAI's newer name for system instructions) failed the request: DeepSeek's encoder
#      raises "Unknown role: developer" -> rendered as a system turn (serve_ds41.py to_encoder_messages).
#   2. Anthropic parallel tool results came out in the order the client sent them: the adapter dropped tool_use.id and
#      tool_result.tool_use_id, so the encoder could not sort them back into call order - and DSML results carry no
#      ids, so the model paired each result with the wrong call.  The adapter keeps the ids now (frontend.py).
#   3. a DSML tag inside a parameter value (a file the model writes about DSML) broke the call: the wrappers
#      <｜DSML｜ calls> / </｜DSML｜ calls> were dropped from the whole stream, values included, and a value's
#      </｜DSML｜ invoke> ended the call.  DeepSeek's encoder writes values raw and its parser ends a value only at
#      </｜DSML｜ parameter>; serve_ds41.py's call end and wrapper filter now skip values the same way.
# Idempotent.
import os
import sys

X = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/xolotl"


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


edit(os.path.join(X, "tools/v41/serve/serve_ds41.py"), [
("""    for m in messages:
        m = dict(m)
        m["content"] = _text(m.get("content"))
""",
"""    for m in messages:
        m = dict(m)
        m["content"] = _text(m.get("content"))
        if m.get("role") == "developer":   # larry (patch-ds41conf.py): OpenAI's newer name for the system turn
            m["role"] = "system"
"""),
("""def ds41_call_end(text: str) -> int:
    return text.find(INVOKE_CLOSE)
""",
"""_P_HEAD_RE = re.compile(r' name="(.*?)" string="(true|false)">', re.DOTALL)


def ds41_call_end(text: str) -> int:
    \"\"\"The call's </｜DSML｜ invoke> - larry (patch-ds41conf.py): the first one OUTSIDE a parameter's value.  DeepSeek's
    encoder writes values raw (a value may hold any DSML tag) and its parser ends a value only at </｜DSML｜ parameter>.
    -1: not complete yet.\"\"\"
    i = 0
    while True:
        e = text.find(INVOKE_CLOSE, i)
        p = text.find(P_OPEN, i)
        if p < 0 or 0 <= e < p:
            return e
        m = _P_HEAD_RE.match(text, p + len(P_OPEN))
        if not m:                     # a head still arriving, or not a parameter head
            if e < 0:
                return -1
            i = p + len(P_OPEN)
            continue
        c = text.find(P_CLOSE, m.end())
        if c < 0:
            return -1                 # inside a value
        i = c + len(P_CLOSE)
"""),
("""    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body
    self.dsml_hold = ""
""",
"""    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body
    self.dsml_hold = ""
    self.dsml_value = False            # larry (patch-ds41conf.py): inside a parameter's value
"""),
("""def _strip_wrappers(self, delta: str, final: bool) -> str:
    \"\"\"Drop <｜DSML｜ calls> / </｜DSML｜ calls> from the stream, holding a partial one until it is decided.\"\"\"
    buf = self.dsml_hold + delta
    for w in WRAPPERS:
        buf = buf.replace(w, "")
    keep = 0 if final else self._hold(buf, WRAPPERS)
    self.dsml_hold = buf[len(buf) - keep:] if keep else ""
    return buf[:len(buf) - keep] if keep else buf
""",
"""def _strip_wrappers(self, delta: str, final: bool) -> str:
    \"\"\"Drop <｜DSML｜ calls> / </｜DSML｜ calls> from the stream, holding a partial one until it is decided.  larry
    (patch-ds41conf.py): not inside a parameter's value, which runs from its head's `">` to the next
    </｜DSML｜ parameter> (DeepSeek's encoder writes values raw: a file the model writes may hold any DSML tag).\"\"\"
    buf = self.dsml_hold + delta
    self.dsml_hold = ""
    out, i, n = [], 0, len(buf)
    while i < n:
        if self.dsml_value:
            c = buf.find(P_CLOSE, i)
            if c < 0:
                keep = 0 if final else self._hold(buf[i:], (P_CLOSE,))
                out.append(buf[i:n - keep])
                self.dsml_hold = buf[n - keep:]
                break
            out.append(buf[i:c + len(P_CLOSE)])
            i, self.dsml_value = c + len(P_CLOSE), False
            continue
        hits = [(j, t) for t in WRAPPERS + (P_OPEN,) for j in (buf.find(t, i),) if j >= 0]
        if not hits:
            keep = 0 if final else self._hold(buf[i:], WRAPPERS + (P_OPEN,))
            out.append(buf[i:n - keep])
            self.dsml_hold = buf[n - keep:]
            break
        j, t = min(hits)
        out.append(buf[i:j])
        if t != P_OPEN:                    # a wrapper outside every value: dropped
            i = j + len(t)
            continue
        m = _P_HEAD_RE.match(buf, j + len(P_OPEN))
        if m:                              # a value starts after the head
            out.append(buf[j:m.end()])
            i, self.dsml_value = m.end(), True
            continue
        if not final and '">' not in buf[j:] and n - j < 1024:   # the head is still arriving
            self.dsml_hold = buf[j:]
            break
        out.append(buf[j:j + len(P_OPEN)])   # not a parameter head: text
        i = j + len(P_OPEN)
    return "".join(out)
"""),
], "patch-ds41conf.py")

edit(os.path.join(X, "serve/frontend.py"), [
("""            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}})""",
"""            elif kind == "tool_use":   # larry (patch-ds41conf.py): with its id (an encoder orders results by it)
                calls.append({"id": block.get("id"),
                              "function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}})"""),
("""                messages.append({"role": "tool", "content": _parts_of(_check_parts(
                    block.get("content"), f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS))})""",
"""                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": _parts_of(_check_parts(
                    block.get("content"), f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS))})"""),
], "patch-ds41conf.py")

edit(os.path.join(X, "serve/test_server.py"), [(
"""        self.assertEqual(anthropic_to_messages({"messages": msgs})[0][-1], {"role": "tool", "content": "r"})""",
"""        # (larry, patch-ds41conf.py: with the result's call id, which an encoder orders parallel results by)
        self.assertEqual(anthropic_to_messages({"messages": msgs})[0][-1], {"role": "tool", "tool_call_id": "t1", "content": "r"})""")],
     "patch-ds41conf.py")
