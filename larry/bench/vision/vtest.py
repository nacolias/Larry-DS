#!/usr/bin/env python3
"""vtest.py PORT OUTDIR - larry, 2026-10-26: pictures through the DeepSeek-V4.1 server: draws two test cards of one size
(different content), asks about the first (OpenAI and Anthropic APIs), follows up on it (the conversation cache should
reuse the picture), asks about the second with the same text (reuse must stop at the picture), and a text-only question.
Prints each answer's head and the engine's reuse line for it (from the engine log given as $VTEST_LOG)."""
import base64
import io
import json
import os
import sys
import time
import urllib.request

from PIL import Image, ImageDraw

PORT, OUT = sys.argv[1], sys.argv[2]
LOG = os.environ.get("VTEST_LOG", "")
KEY = os.environ.get("VTEST_KEY", "")
os.makedirs(OUT, exist_ok=True)


def card(kind):
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    if kind == 1:
        d.rectangle([40, 40, 360, 280], fill=(220, 30, 30))
        d.ellipse([440, 60, 760, 300], fill=(30, 90, 220))
        d.polygon([(400, 560), (250, 330), (550, 330)], fill=(30, 170, 60))
        d.text((60, 320), "LARRY 42", fill="black")
    else:
        d.ellipse([60, 60, 380, 380], fill=(250, 200, 0))
        d.rectangle([450, 300, 760, 560], fill=(120, 40, 160))
        d.text((480, 120), "BLUE MOON", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(path, body):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(f"http://{os.environ.get('VTEST_HOST', '172.23.0.1')}:{PORT}{path}", json.dumps(body).encode(), h)
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r, time.time() - t0


def last_reuse():
    if not LOG or not os.path.exists(LOG):
        return ""
    lines = [l for l in open(LOG, errors="replace") if l.startswith("serve: reuse ") or l.startswith("serve: prompt ")]
    return " | ".join(l.strip()[:150] for l in lines[-2:])


Q = "Describe this image: which shapes and colours do you see, and what text does it contain? Be brief."
c1, c2 = card(1), card(2)
oa = lambda b64, q: [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                                  {"type": "text", "text": q}]}]
think_off = {"chat_template_kwargs": {"enable_thinking": False}}

r, s = post("/v1/chat/completions", {"model": "x", "messages": oa(c1, Q), "max_tokens": 160, "temperature": 0, **think_off})
a1 = r["choices"][0]["message"]["content"]
print("1 OpenAI card 1 (%.1f s): %s" % (s, a1.replace("\n", " ")[:400]))
print("   ", last_reuse())

msgs = oa(c1, Q) + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What number is written there?"}]
r, s = post("/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 60, "temperature": 0, **think_off})
print("2 follow-up (%.1f s): %s" % (s, r["choices"][0]["message"]["content"].replace("\n", " ")[:300]))
print("   ", last_reuse())

r, s = post("/v1/chat/completions", {"model": "x", "messages": oa(c2, Q), "max_tokens": 160, "temperature": 0, **think_off})
print("3 OpenAI card 2, same text (%.1f s): %s" % (s, r["choices"][0]["message"]["content"].replace("\n", " ")[:400]))
print("   ", last_reuse())

r, s = post("/v1/messages", {"model": "x", "max_tokens": 160, "temperature": 0, "thinking": {"type": "disabled"},
                             "messages": [{"role": "user", "content": [
                                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": c1}},
                                 {"type": "text", "text": Q}]}]})
txt = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
print("4 Anthropic card 1 (%.1f s): %s" % (s, txt.replace("\n", " ")[:400]))
print("   ", last_reuse())

r, s = post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "What is the capital of Norway? One word."}],
                                     "max_tokens": 20, "temperature": 0, **think_off})
print("5 text only (%.1f s): %s" % (s, r["choices"][0]["message"]["content"].replace("\n", " ")[:200]))
