#!/usr/bin/env python3
"""venc_test.py OUTDIR [THREADS] - larry, 2026-10-26: one image through Xolotl's DeepSeek-V4.1 vision encoder process
(tools/v41/vision/ds41_vision.py, the strata-vision line protocol) on the CPU, with the sidecar's vision tower and
DeepSeek's reference vision.py / image_processor.py.  Draws a test card (text, shapes, colours), encodes it, and
prints the encoder's reply, the SVE1 record's shape and simple statistics of its rows."""
import os
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

OUT = sys.argv[1]
THREADS = sys.argv[2] if len(sys.argv) > 2 else "6"
D = "/opt/ai/dsv41"
os.makedirs(OUT, exist_ok=True)

img = Image.new("RGB", (800, 600), "white")
d = ImageDraw.Draw(img)
d.rectangle([40, 40, 360, 280], fill=(220, 30, 30))
d.ellipse([440, 60, 760, 300], fill=(30, 90, 220))
d.polygon([(400, 560), (250, 330), (550, 330)], fill=(30, 170, 60))
d.text((60, 320), "LARRY 42", fill="black")
path = os.path.join(OUT, "card.png")
img.save(path)

cmd = [D + "/tvenv/bin/python", "-I", D + "/xolotl/tools/v41/vision/ds41_vision.py",
       "--mmproj", D + "/models/gsq1017/V41-MTP-Vision-1017.Q2_K.gguf",
       "--model", D + "/models/gsq1017/V41-GSQ-1017.gguf",
       "--device", "cpu", "--threads", THREADS,
       "--reference-dir", D + "/qv/ds1017/inference"]
t0 = time.time()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
ready = p.stdout.readline().strip()
print("ready:", ready, "after %.1f s" % (time.time() - t0))
if not ready.startswith("READY"):
    print(p.stderr.read()[-3000:])
    sys.exit(1)
out = os.path.join(OUT, "card.sve")
t1 = time.time()
p.stdin.write(f"ENC {path} {out}\n")
p.stdin.flush()
reply = p.stdout.readline().strip()
print("reply:", reply, "after %.1f s" % (time.time() - t1))
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=60)
if reply.startswith("OK"):
    with open(out, "rb") as f:
        magic, n, nx, ny, ne = struct.unpack("<5i", f.read(20))
        rows = np.frombuffer(f.read(), np.float32).reshape(n, ne)
    print("SVE1 magic %#x n %d nx %d ny %d n_embd %d" % (magic, n, nx, ny, ne))
    print("row norms: first %.3f  middle %.3f  last %.3f  mean %.3f  finite %s" % (
        np.linalg.norm(rows[0]), np.linalg.norm(rows[n // 2]), np.linalg.norm(rows[-1]),
        float(np.linalg.norm(rows, axis=1).mean()), bool(np.isfinite(rows).all())))
else:
    print(p.stderr.read()[-3000:])
