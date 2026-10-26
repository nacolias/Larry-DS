#!/usr/bin/env python3
# readme-night.py - /opt/ai/dsv41/README.md: the 2026-10-21 night section (where the four cards' time goes, spreading
# work over them), the new knobs, the next list.  Idempotent.
import sys, shutil
P = "/opt/ai/dsv41/README.md"
s = open(P).read()
if "## Measured (2026-10-21 night" in s:
    print("already"); sys.exit(0)
shutil.copy(P, P + ".bak-20261021d")

SECTION = """## Measured (2026-10-21 night: where the four cards' time goes, and spreading work over them)

Service after this round (`tsvc2.sh`, commit a54327cb + `STRATA_ENGRAM_THREADS=256`): svcbench 64.5 / 63.8 tok/s, essay
53.3 / 48.4, prompts 8K 963 / 16K 1221 / 31K 1510 tok/s, needles 200-31K found, tools fine, hidden RAM peak 41 GiB -
the same as before (the new paths are off by default).

**Utilization, measured** (one-shot, GPU clocks pinned; STRATA_GPU_TL, STRATA_DEC_PROF, STRATA_XE_PROF, unitrace, xpu-smi):

| | card 0 | cards 1-3 | host (engine thread) |
| --- | --- | --- | --- |
| plain decode, a layer (564 us) | 432 us of kernels (77 %): attention 403 us wall (~350 kernels + gaps between ~38 dependent kernels), shared expert 82, expert wait + sum 63, finish 14 | the routed experts ~57 us (~10 %) + promotion chunks | attention graph submit ~210 us (~6.5 us a ggml op), shexp 36, tier ~65, readback wait ~230 (idle) |
| DSpark verify, a layer (~880 us, 2.3 tokens) | attention 521, shexp 103, its own experts 98 (start 125 us after the readback, behind the shexp) | start 56-63 us after the readback, experts ~158 us (~160 GB/s: the dp4a kernels slow down per token, below) | tier 164 us (card submissions 58, CPU-pool experts 71: 0.2 a call at ~350 us each) |
| 31.6K prompt (22.9 s warm) | 71 % busy, EU active 24 %, 130 GB/s read | 41 % busy, EU 8-11 %, ~14 GB/s, PCIe rx 2.3 GB/s each | the tier's calls 18 s wall (cards 14.6 s, combine 2.2 s); arena copies 63 GB for 60 layer passes |

Single-stream decode keeps card 0's dense half (~6 GB of Q6_K a token: 89 tok/s at 580 GB/s, before any small kernel)
on one card; cards 1-3 can only take the dense half's work if the host keeps ahead of the cards - and it can not:
**every multi-card split is bounded by host submissions** (~4 us a SYCL command, ~6.5 us a ggml op, ~14 us a graph launch
that waits on another card's event), the router readback serializes host and cards once a layer, and the host is already
~60 % busy at ~43 tok/s.

- **The cards themselves split fine** (`bench/tp/tp_probe`, host submissions taken out by a busy-kernel gate): a cross-card
  dependency ~4.5 us a hop, a scatter + reduce round trip with copies 15 us; the attention output (wo_a + wo_b, 62 MB)
  131.6 us on card 0 alone -> 45.6 us over four cards; q_b 52 -> 30.  Host-paced the same splits take 40 / 93 / 55 us.
- **`STRATA_TP_OPROJ`** (commit a54327cb, off): the attention output projection split by head groups over the four cards
  (card k: groups 2k, 2k+1 of wo_a and wo_b's matching 2048 columns, 15.5 MB a card a layer; a partial sum back).  =1 an
  op in the attention graph (`GGML_OP_DSV41_OPROJ`), =2 the kept graph cut in two around it, each expert card's part a
  recorded SYCL graph.  In the engine card 0's projection drops ~130 -> ~79 us, but the host submits ~68 us more a layer:
  plain decode 43.27 -> 44.80 tok/s (+3.5 %, =2), DSpark verify passes unchanged (34.3-34.9 vs 33.9-34.9 ms/pass).  Needs
  a 4.2 GiB expert-card reserve.  Decode-path PPL in the noise.  The old `STRATA_HEAD_SPLIT_MB=30` loss (-12 %) was the
  same thing: the host reached the tier 64 us after each readback (8 normally), so the cards started late.
- **SYCL graph replay of the kept graphs re-tested** (`GGML_SYCL_GRAPH_CACHE=1`): correct (decode-path PPL 1.9833 vs eager
  1.9899 / 1.9911; "changes tokens" was the eager path's own nondeterminism - two eager runs part at token 32), and it cuts
  the attention graph's host submit 213 -> 26 us a layer, but replayed verify graphs run slower on the GPU (attention 516
  -> 580 us) and plain decode does not move: still off.
- **The dp4a expert kernels slow down with a job's tokens** (`test_ds41_xe --bench-q8`, 24 experts rotated past the L2):
  3 experts 66 / 88 / 116 / 170 us at 1 / 2 / 3 / 5 tokens (425 -> 160 GB/s) - every weight row re-reads every token's
  activations.  The armed sequences also pick the kernel template from the pass width, not the jobs: 1-token jobs in a
  3-5 token pass run 10-25 % slower.
- **`STRATA_XE_XMX`** (commit a54327cb, off): the decode / verify experts on XMX (fp16 DPAS, `xe_xmx.cpp`): flat in tokens
  (~130 us for 3 experts at 1-5 tokens) and 30x closer to the CPU reference (2e-4..8e-4 vs 5e-3..1.3e-2), but ~210 GB/s
  - the fp16 decode and the per-element fragment writes cost more than the dp4a kernels' whole job below 4 tokens.  The
  int8 route (B through local memory, verified: `bench/tp/jm8_slm.cpp`) is the next try.
- **Engram reader pool 32 -> 256** (`STRATA_ENGRAM_THREADS`, service): a cold 31.6K prompt's first-batch wait for its rows
  1.8 -> 0.4 s (the pool is the queue depth over both NVMe drives); the prompt 23.4 -> 23.0 s.
- **One 32K prompt batch does not fit**: the expert cards' GEMM scratch for ~16K-token calls (0.84 GiB free at the 3.5 GiB
  reserve) - "tier run refused", and the engine then hung in its teardown.  Two 16K batches with enc-skip stay.
- Not the cause of anything (ruled out): ASPM L1 (policy performance), GT C6 (forcewake held), different SYCL contexts
  (one platform default context everywhere), VMM pool memory for card-to-card copies (no difference in the probe).

"""

anchor = "## Measured (2026-10-21 evening: prompt stalls, the DSpark draft, the verify expert phase)"
if s.count(anchor) != 1: sys.exit("anchor")
s = s.replace(anchor, SECTION + anchor)

knob_anchor = "| `STRATA_DSPARK_DENSE_TYPE` | q6_K (service) | the draft's Q8_0 projections requantized on load (engine default: off); `STRATA_DSPARK_PROF=1` the draft's host phases |"
knobs = knob_anchor + """
| `STRATA_ENGRAM_THREADS` | 256 (service) | the Engram row reader's pool = its queue depth over both NVMe drives (engine default 32) |
| `STRATA_TP_OPROJ` | 0 | 1 / 2: the attention output projection split over the four cards (in-graph op / the attention graph in two halves); needs `STRATA_XE_RESERVE_GIB` 4.2; `STRATA_TP_GRAPH` (1) records each card's part, `STRATA_TP_PROF=1` its host cost |
| `STRATA_XE_XMX` | 0 | 1: decode / verify experts on XMX (flat in tokens, slower below ~4 tokens); `STRATA_XMX_KSG` / `_KSD` / `_GRF` its K splits / register mode |"""
if s.count(knob_anchor) != 1: sys.exit("knob anchor")
s = s.replace(knob_anchor, knobs)

next_anchor = "## Not done yet / next\n\n"
nxt = next_anchor + """- 2026-10-21 night: done - utilization accounting, the attention output split, XMX experts (both opt-in), Engram pool.
  Next, in order of expected gain for the service (DSpark): a GPU-side expert dispatch (no host round trip between the
  router and the cards - the cards start ~60 us after a verify readback now, and it lets the host run ahead, which the
  splits need); int8 XMX experts (flat in tokens at dp4a speed: wider verify passes then cost little); the shared expert
  on cards 1-3 riding along with the routed experts; then more of the dense half split (q_b) once the host is ahead.
"""
if s.count(next_anchor) != 1: sys.exit("next anchor")
s = s.replace(next_anchor, nxt)
open(P, "w").write(s)
print("updated")
