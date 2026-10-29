#!/usr/bin/env python3
# svc-loopwrap.py [CONFIG] - the service (2026-10-29): "reasoning_loop_recovery": "wrap" in larry-ds41.json
# (patch-loopwrap.py): thinking that repeats whole passages is closed the way the thinking budget closes it and the model
# answers - the agent session of 2026-10-29 05:39-05:59 had nine loops of ~24,600 tokens run into the 24,576-token budget
# (~2 min each), and each loop's text stayed in the conversation.  Backup larry-ds41.json.pre-loopwrap-20261029; the file
# is replaced by a rename.  The server reads it at start (larry-mode restart).  Idempotent.
import json
import os
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/larry-ds41.json"
c = json.load(open(P))
if c.get("reasoning_loop_recovery") == "wrap":
    print("already set")
    sys.exit(0)
out = {}
for k, v in c.items():
    if k != "reasoning_loop_recovery":
        out[k] = v
    if k == "reasoning_budget_tokens":
        out["reasoning_loop_recovery"] = "wrap"
out.setdefault("reasoning_loop_recovery", "wrap")
shutil.copy2(P, P + ".pre-loopwrap-20261029")
tmp = P + ".new"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
shutil.copymode(P, tmp)
os.rename(tmp, P)
print("set", P)
