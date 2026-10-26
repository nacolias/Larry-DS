#!/usr/bin/env python3
# readme-update.py - /opt/ai/dsv41/README.md: the 2026-10-21 evening section, new knobs, the corrected open problem.
import sys, shutil
P = "/opt/ai/dsv41/README.md"
s = open(P).read()
if "## Measured (2026-10-21 evening" in s:
    print("already"); sys.exit(0)
shutil.copy(P, P + ".bak-20261021c")

SECTION = """## Measured (2026-10-21 evening: prompt stalls, the DSpark draft, the verify expert phase)

Service (`tsvc2.sh`; the unit pins the GPU clocks - one-shot runs do NOT unless the script calls `larry-gpu-clock pin`):

| | goal start (2026-10-19) | 2026-10-21 afternoon | now |
| --- | --- | --- | --- |
| svcbench | 20.6 tok/s | 58.5-61.1 | 62.6-67.0 (DSpark cut 0.6), 65.1-65.3 (0.3) |
| essay (3 runs) | 17.3 | 44.8-46.4 | 49.7-51.8 (0.6), 49.5-52.1 (0.3) |
| prompt 8K / 16K / 31K | ~310 / - / ~440 | 913 / 1047 / 1205 | 964 / 1169 / 1509 tok/s |

Needles 200-31K found, tools fine, hidden RAM peak 41 GiB.

- **Prompt** (commit b0cc030c; full-prompt unitrace + `STRATA_ENG_PROF`): one-shot 31.6K 26.3 -> 22.1 s (warm Engram
  pages; 26.5 -> 23.5 s with a dropped page cache), outputs identical.
  - wo_a reordered per group slice at load (`STRATA_PREREORDER_SLICES`): before the first decode token ggml-sycl's
    generic path gathered each group's permuted activations with 2-D dpct copies at ~45 GB/s (11 % of card 0's time).
  - fp16 weight cache for the prompt GEMMs, wo_a's slices included (`GGML_SYCL_LARRY_F16CACHE_MB` 768, native buffers,
    `_FREE_MB` 3072 guard, released before decode's VRAM seed). Its fallback crashed the service once on a 34-token
    prompt (pool freed out of stack order) - fixed; `GGML_SYCL_LARRY_F16CACHE_FREE_MB=1000000` forces the fallback.
  - Engram: a prompt batch's rows land in pinned memory (no 0.8 GB memset) and go up async (`STRATA_ENG_PIN`); the next
    batch's rows are read into the row cache once this batch's reads are done (`STRATA_ENG_PREFETCH`). Before: a 16K
    batch's 0.8 GB pageable upload took ~0.45 s under the submit lock and batch 2's reads stalled every card.
  - Left: batch 1's cold Engram reads (~1.4M random 4 KiB reads, ~2 s with an empty page cache) and ~1 s of
    all-idle time at the prompt's start / batch boundary; card 0 is ~60 % kernel-busy over the prompt.
- **DSpark draft** (commit 119d9d4d): 7.7 -> 4.5 ms a pass. It was host-bound: ggml-sycl's mul_mat_id read the routing
  ids back and drained the queue 9 times a draft, then ran a generic Q2_K mat-vec per expert at 28-49 GB/s.
  `GGML_SYCL_LARRY_MMID_Q2K` (all tokens x slots in one dp4a launch, ids on the device; `_CHECK=1` compares),
  `GGML_SYCL_LARRY_ARGMAX` (the 129K-logit argmax over 64 work-groups: 60 -> ~7 us), `STRATA_DSPARK_DENSE_TYPE=q6_K`
  (service: the draft's Q8_0 projections on fastmv). Acceptance unchanged.
- **XMX multi-column mat-vec** (commit 61124278, `GGML_SYCL_LARRY_XMXMV`, off): exact to 1e-4 vs dp4a, 8-17 us faster
  on shexp down / q_a in isolation (`bench/dqb/xmx_mv.cpp`; `lpr_bw.cpp`: one-row-per-lane 16-byte loads stream at
  480-560 GB/s), no end-to-end gain.
- **The verify expert phase - the "late sum" corrected.** At full verify width (`STRATA_DSPARK_CONF=0`, ~5.5 tokens a
  pass, 40.8 ms) the expert cards really compute ~205 us a layer (~100 MB each, bandwidth-bound); the timeline's "copy
  done -> sum" averages mix in calls that wait for the CPU part or card 0's experts. Debug knobs (wrong output, timing
  only): dropping the armed cards' dependency 40.8 -> 34.4 ms/pass, also the CPU and card-0 parts 29.1. GPU timestamps
  (unitrace, unarmed): the sum starts a median 4.5 us after its last part. Card 0 pulling the parts (P2P reads) instead
  of the cards pushing them: no change. So what is left there is real work and the host release (~42 us readback ->
  card start; a GPU-side dispatch would save ~20-25 us a layer) plus the CPU-pool experts (~40 us a layer on average).
- **DSpark's confidence cut 0.6 -> 0.3** (`STRATA_DSPARK_CONF` in `ds41-xe-engine.sh`): with a 4.5 ms draft, wider verify
  passes pay. One-shot sweep, two prompts x2-4 runs, tok/s: 0.1 55.1 / 69.3, 0.2 56.3 / 75.9, 0.3 58.4 / 75.3,
  0.45 58.4 / 72.3, 0.6 54.3 / 73.0, 0.75 50.8 / 73.3 (essay / p1). Service: 2.35 tokens a pass (was 1.88),
  svcbench 65.1-65.3 (steadier), essay unchanged.
"""

anchor = "## Measured (2026-10-21 afternoon: fused hyper-connections, and where the time goes now)"
if s.count(anchor) != 1: sys.exit("anchor")
s = s.replace(anchor, SECTION + "\n" + anchor)

old_open = "dependencies met 249 us later, sum 9. **Open problem**: the dense queue sees the expert cards' parts 25-30 us (decode)"
new_open = "dependencies met 249 us later, sum 9. **Open problem** (resolved for verify passes in the evening section: real expert work, not a lag): the dense queue sees the expert cards' parts 25-30 us (decode)"
if s.count(old_open) == 1: s = s.replace(old_open, new_open)

knob_anchor = "| `STRATA_DENSE_REQUANT` | q6_K | empty = keep the dense Q8_0 (slower decode, 1.7 GB more VRAM) |"
knobs = knob_anchor + """
| `STRATA_PREREORDER_SLICES` | 1 | wo_a reordered per group slice at load: every prompt span takes the per-slice GEMM |
| `GGML_SYCL_LARRY_F16CACHE_MB` / `_FREE_MB` | 768 / 3072 | prompt GEMMs keep fp16 copies of their weights across a batch's spans (0 = off) / VRAM kept free |
| `STRATA_HC_FUSE_MAXN` | 8 | the fused hyper-connection ops for passes of at most this many tokens (prompt spans were slower fused) |
| `STRATA_ENG_PIN`, `STRATA_ENG_PREFETCH` | 1 | prompt Engram rows in pinned memory + async upload; the next batch's rows read ahead (`STRATA_ENG_PROF=1` prints the phases) |
| `GGML_SYCL_LARRY_MMID_Q2K`, `GGML_SYCL_LARRY_ARGMAX` | 1 | DSpark draft: device-side Q2_K experts for 2-8 tokens; argmax of long rows over 64 work-groups (`_CHECK=1` compare) |
| `STRATA_DSPARK_DENSE_TYPE` | q6_K (service) | the draft's Q8_0 projections requantized on load (engine default: off); `STRATA_DSPARK_PROF=1` the draft's host phases |
| `GGML_SYCL_LARRY_XMXMV` | 0 | 1: XMX multi-column Q6_K mat-vecs for verify passes (no end-to-end gain measured) |"""
if s.count(knob_anchor) != 1: sys.exit("knob anchor")
s = s.replace(knob_anchor, knobs)

next_anchor = "## Not done yet / next\n\n"
nxt = next_anchor + """- 2026-10-21 evening: done - prompt stalls (wo_a, fp16 weight cache, Engram), the DSpark draft. Next: a GPU-side expert
  dispatch (no host round trip between the router and the cards' start: ~20-25 us of every decode / verify layer),
  the CPU-pool experts in verify (~40 us a layer), the draft's remaining unfused HC / shared expert (~0.3 ms a pass),
  batch 1's cold Engram reads in long prompts.
"""
if s.count(next_anchor) != 1: sys.exit("next anchor")
s = s.replace(next_anchor, nxt)
open(P, "w").write(s)
print("updated")
