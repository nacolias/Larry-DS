#!/usr/bin/env python3
"""imagetest.py - a picture through the ds41 server (ds41api.py): a drawn test card (three shapes, "LARRY 42") as an
OpenAI image_url part, then a follow-up on the same picture (the conversation cache keeps it).  Needs pillow: tsvc2.sh
runs it with the image encoder's torch venv (DS41_TVENV)."""
import base64, io, sys, time, urllib.error
import ds41api
from PIL import Image, ImageDraw
img = Image.new("RGB", (800, 600), "white")
d = ImageDraw.Draw(img)
d.rectangle([40, 40, 360, 280], fill=(220, 30, 30))
d.ellipse([440, 60, 760, 300], fill=(30, 90, 220))
d.polygon([(400, 560), (250, 330), (550, 330)], fill=(30, 170, 60))
d.text((60, 320), "LARRY 42", fill="black")
buf = io.BytesIO()
img.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
def ask(msgs, n):
    t = time.time()
    d = ds41api.chat({"model": "larry", "messages": msgs, "max_tokens": n, "temperature": 0, "reasoning_effort": "none"},
                     timeout=600)
    return (d["choices"][0]["message"].get("content") or "").replace("\n", " "), time.time() - t
msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": "Which shapes and colours does this image show, and what "
                                                              "text? Be brief."}]}]
try:
    a1, t1 = ask(msgs, 160)
    print(f"picture ({t1:.1f} s): {a1[:300]!r}")
    msgs += [{"role": "assistant", "content": a1}, {"role": "user", "content": "What number is written there? Just the number."}]
    a2, t2 = ask(msgs, 20)
except urllib.error.HTTPError as e:
    print(f"picture: HTTP {e.code} {e.read()[:300]!r}")
    sys.exit(1)
ok = "42" in a1 and "42" in a2
print(f"follow-up ({t2:.1f} s): {a2[:80]!r} - {'right' if ok else 'WRONG'}")
sys.exit(0 if ok else 1)
