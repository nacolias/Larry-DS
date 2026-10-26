#!/usr/bin/env python3
# patch-vision-server.py [XOLOTL_DIR] - larry, 2026-10-26: pictures through the DeepSeek-V4.1 server (with
# patch-vision-engine.py on the engine side and a "vision" section in the config: the encoder process).
#   serve/server.py      IMAGE_NEEDS_START (default True = as before): Service.prepare expands a picture's image token
#                        into one per embedding row only after <|vision_start|> (Qwen's template writes the pair).
#                        DeepSeek's encoder writes <｜deepseek_image｜> alone, and patch-literals.py already keeps that
#                        text in a message plain, so with False every image token is a picture.
#   serve_ds41.py        S.IMAGE_PAD = <｜deepseek_image｜>, IMAGE_NEEDS_START = False; a message's picture parts reach
#                        DeepSeek's encoder as image blocks (they were dropped: only the text parts were kept); the
#                        server's images_of in the order the encoder writes the placeholders (a run of tool results in
#                        its assistant message's call order, as the encoder sorts them).
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


edit(os.path.join(X, "serve/server.py"), [
("""IMAGE_PAD = "<|image_pad|>"
""",
"""IMAGE_PAD = "<|image_pad|>"
# larry (patch-vision-server.py): a picture's image token is one only right after VISION_START (Qwen's template writes the
# pair); False for a model whose template writes the image token alone (DeepSeek-V4.1's <｜deepseek_image｜>)
IMAGE_NEEDS_START = True
"""),
("""                if t == pad and j > 0 and ids[j - 1] == start:""",
"""                if t == pad and (not IMAGE_NEEDS_START or (j > 0 and ids[j - 1] == start)):"""),
], "patch-vision-server.py")

edit(os.path.join(X, "tools/v41/serve/serve_ds41.py"), [
("""def to_encoder_messages(messages, tools):""",
"""def _content(content):
    \"\"\"larry (patch-vision-server.py): a message's content for DeepSeek's encoder - its text as before, or, when it has
    pictures, its text and picture parts in order (the encoder writes <｜deepseek_image｜> for each picture)\"\"\"
    if not (isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image" for p in content)):
        return _text(content)
    out = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "image":
            out.append({"type": "image", "source": p.get("source")})
        elif p.get("type") in ("text", "input_text", None) and isinstance(p.get("text"), str):
            out.append({"type": "text", "text": p["text"]})
    return out


def ds41_images_of(messages):
    \"\"\"larry (patch-vision-server.py): the server's picture sources in the order DeepSeek's encoder writes their
    placeholders - message order, except that a run of tool results follows its assistant message's call order (the
    encoder sorts parallel results by call id; ids it does not know sort first, stably, as its sorted() does).\"\"\"
    out, order, i = [], {}, 0

    def pics(m):
        c = m.get("content")
        return [p["source"] for p in c if isinstance(p, dict) and p.get("type") == "image"] if isinstance(c, list) else []
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            order = {c.get("id"): k for k, c in enumerate(m["tool_calls"]) if c.get("id")}
        if m.get("role") == "tool":
            j = i
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            for t in sorted(messages[i:j], key=lambda t: order.get(t.get("tool_call_id"), 0)):
                out += pics(t)
            i = j
            continue
        out += pics(m)
        i += 1
    return out


def to_encoder_messages(messages, tools):"""),
("""        m["content"] = _text(m.get("content"))""",
"""        m["content"] = _content(m.get("content"))   # (larry, patch-vision-server.py: pictures kept)"""),
("""S.IM_END = E.eos_token
""",
"""S.IM_END = E.eos_token
# larry (patch-vision-server.py): DeepSeek's picture marker, written alone by its encoder; the pictures in its order
S.IMAGE_PAD = E.IMAGE_PLACEHOLDER
S.IMAGE_NEEDS_START = False
S.images_of = ds41_images_of
"""),
], "patch-vision-server.py")
