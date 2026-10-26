#!/usr/bin/env python3
# patch-engtidy.py - ds41_dense.cpp: the Engram helpers patch-engprof / engpf / engpin added go ABOVE the original
# "Wait for the pass's Engram reads" comment, which again sits on engram_wait_host / engram_ready.  Idempotent.
import sys
D = "/opt/ai/dsv41/xolotl/tools/v41/engine/ds41_dense.cpp"
s = open(D).read()
doc = """// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
// larry: the Engram batch's host-side wait alone (no device work: a pipelined span runs it before the submit lock)
"""
a = s.find(doc)
if a < 0: sys.exit("doc anchor")
if s[a + len(doc):].startswith("static bool engram_wait_host("):
    print("already"); sys.exit(0)
prof_start = s.find("// STRATA_ENG_PROF: a prompt batch's Engram phases", a)
wh = s.find("static bool engram_wait_host(Ds41Dense::Impl & im) {", a)
wh_end = s.find("\n}\n", wh) + 3
pin = s.find("// larry: the pinned rows of a prompt batch of `per` floats a slot", wh_end)
er = s.find("static bool engram_ready(Ds41Dense::Impl & im) {", wh_end)
if not (a < prof_start < wh < wh_end <= pin < er): sys.exit(f"layout {a} {prof_start} {wh} {wh_end} {pin} {er}")
helpers = s[prof_start:wh]            # prof statics + eng_pf_launch
wait_host = s[wh:wh_end]
pin_ok = s[pin:er]                    # eng_pin_ok (+ trailing blank line)
new = helpers + pin_ok + doc + wait_host + "\n"
s = s[:a] + new + s[er:]
open(D, "w").write(s)
print("tidied")
