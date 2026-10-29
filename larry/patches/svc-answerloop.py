#!/usr/bin/env python3
# svc-answerloop.py [CONFIG] - the service (2026-10-29): "answer_loop_stop": 0.9 in larry-ds41.json (patch-answerloop.py):
# an answer whose last 2,000 words lie 90 % or more in 12-word passages seen three times (or ends in a short-period run)
# is ended there ("length") - a looping answer otherwise runs to max_tokens.  Backup
# larry-ds41.json.pre-answerloop-20261029; the file is replaced by a rename.  The server reads it at start.  Idempotent.
import json
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/larry-ds41.json"
c = json.load(open(P))
if c.get("answer_loop_stop") == 0.9:
    print("already set")
    sys.exit(0)
out = {}
for k, v in c.items():
    if k != "answer_loop_stop":
        out[k] = v
    if k == "reasoning_loop_recovery":
        out["answer_loop_stop"] = 0.9
out.setdefault("answer_loop_stop", 0.9)
shutil.copy2(P, P + ".pre-answerloop-20261029")
tmp = P + ".new"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("set", P)
