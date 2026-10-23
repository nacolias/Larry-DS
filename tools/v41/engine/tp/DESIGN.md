# engine1 - DeepSeek-V4.1 decode / verify on N cards (tensor parallel), host out of the loop

Status: phases A-D done; in the engine behind STRATA_TP=N (default 0), in the service since 2026-10-23 02:35
(STRATA_TP=4), the pager (STRATA_TP_PAGE=1) since 06:02, promotions in time for the next pass since 17:31, the verify
pass's kernel round since 20:35 - see the status sections at the end.  User go-ahead
2026-10-22 ("option 3": the four-card engine, TP degree a setting, any card count).  Mirror of
larry:/opt/ai/dsv41/xolotl/tools/v41/engine/tp/ (this file is DESIGN.md there).

## Why

Step 0 (bench/engine0/tp_layer.cpp, delay kernels for the small ops): TP over 4 cards with host-free execution 355-378
us a decode layer vs 474-504 for today's layout; host-paced TP 925 (the host cannot feed 4 cards).  Today's engine after
the fusion round: 536 us a decode layer on the GPU timeline (card 0 dense ~360 + post-readback ~175).  Target:
~250-300 us a layer at N = 4 (plain decode ~46 -> ~80 tok/s; verify passes similar).

## Scope (first version)

* Decode / verify passes only (n <= 7 tokens: the fused attention's widths).  The prompt path (layer-major batches,
  DSpark draft, park / snapshot, the server protocol) stays in today's engine; the TP core takes over the per-pass
  layer loop (begin -> 40 x {attn_router, tier, finish} -> logits) when STRATA_TP=N (0 = today's path).
* N = 1..4 cards (ranks 0..N-1 = the first N GPUs; rank 0 = today's dense card).  Nothing assumes 4: splits are
  computed from N and the shapes; uneven splits allowed (N = 3: o-groups 3/3/2).
* One SYCL context: the platform default context (ggml-sycl and the xe tier use it: card-to-card copies between their
  allocations work).

## The split (rank k of N)

| part | how | per card at N = 4 |
| --- | --- | --- |
| hc mixes / norms / q_a+kv / kv_fin / compressor / indexer / top-k / router / sinks | replicated (same kernels, same inputs -> bit-identical on every card) | 13 MB a layer (+ indexer 10.5 MB on 8 layers) |
| q_b | rows of its heads (whole o-groups) | 8.6 MB |
| attention | its heads (attn_dec, XHG 16 or 32, head tail masked) | 16 heads |
| wo_a | its o-groups (8 heads each) | 2 groups, 6.9 MB |
| wo_b | the K columns of its groups -> partial sum | 8.6 MB |
| shared expert | n_ff in 256-blocks (9): gate/up rows + down K columns -> partial | 3/2/2/2 blocks, 7.3 MB |
| routed experts | each card the selected experts it holds -> same partial | ~1.5 of 6 |
| CPU-pool experts | host thread; rank 0 adds them (waits only on layers that have one) | |
| Engram kv (layers 1, 14) | output rows -> all-gather | 32 MB |
| LM head | vocab rows -> logits slices to the host | 136 MB |

Weights: every rank owns its shard copies (GGUF Q8_0 -> Q6_K per WHOLE row, as the dense loader does: bit-identical
weights; then block-sliced; planar-reordered on the host).  Card 0 keeps ggml's full copy for the prompt path
(~2 GB of card-0 VRAM extra at N = 4; sharing ggml's planes via offsets is a later saving).

## Cross-card sums

Two all-reduces a layer (attention output, FFN output: n x 5120 floats).  Each card copies its partial into slot k of
every peer (queue memcpy = copy engine P2P), then a sequence number into the peer's flag k; each card spins (bounded)
on its N-1 flags; the CONSUMER kernel (hc_fab) reads the N slots and sums them in rank order 0..N-1 (same association
on every card -> replicated state stays bit-identical).  Sequence numbers = pass counter * 4096 + sync index (a
device counter bumped by each card's first kernel of the pass).

## Host-free execution

Per (card, n, cap bucket) a SYCL graph of the whole pass (40 layers), replayed: N graph launches a pass.  Everything
that changes per pass is device memory written before the launch (one small H2D a card or pinned reads):
positions, window maps, slot maps, nvis, lastblk, embeddings, the Engram rows.  cap (compressed rows scanned by the
indexer) = next_pow2(visible) as today: graphs per bucket, recorded lazily.

Host gates (GPU waits for the host, no submission at release): a spin kernel on a pinned-memory flag with
system-scope acquire loads (hs_probe B1: seen in 2.9 us; RE-VERIFY in graphs).  Used for: the Engram rows (layers 1 /
14: read from disk while layer 0 runs), the CPU-pool experts (only on layers whose routing has one: the route kernel
sets need_cpu[l], the gate kernel returns at once otherwise).  GPU -> host: rank 0 copies ids / wts + a sequence word
into a pinned mailbox after each router; a host thread polls it, runs the CPU experts, writes their sum + the flag.

## State

Rank 0 uses today's state tensors on card 0 (rings, compressed KV, indexer keys, ratio-2 compressor state, x / pre
state columns) through their device pointers: the prompt path, DSpark and park / snapshot see what decode wrote.
Ranks 1..N-1 hold replicas, synced from card 0 (card-to-card copies) after anything else wrote the state: a prompt
batch (its position range), restore / unpark (rings + rows), set_columns.  KV is small (MQA, 4 kv sources): ~0.3 GB at
100K tokens.

## Experts

Residency: the xe tier's placement (cards' slabs + flex slots).  Per card a device table [layer][expert] -> job
template (blob pointer, types, offsets) or -1; rebuilt by the host when residency changes (between passes).  Route
kernel per card: its jobs (tokens of the same expert merged), nj in device memory; expert kernels launched for the
pass's maximum job count, empty slots exit (graph-replay friendly).  Promotion copies: kernels reading pinned memory
(an H2D copy >= 128 KB delays the next kernel ~300 us).

## Phases

A. tp_bench: real dense weights from the GGUF on N cards, synthetic KV state / experts, the whole layer stack
   host-free; us/layer for N = 1..4 and n = 1..5; numerics N vs 1 (only split sums differ).
B. Numerics vs today's engine (decode-path logits KL, PPL) once integrated.
C. Integration: STRATA_TP=N in ds41_generate's decode / verify paths, state sharing, expert tables, CPU pool, logits.
D. Service validation (tsvc2, svcbench / essay, needles / tools), N = 2 / 3 / 4, README, commit.

## Status 2026-10-23 ~01:00 (phases A-C done, D blocked on VRAM; committed off by default)

Built: libds41_tp.so (tools/v41/engine/tp/), ds41_generate_xe STRATA_TP=N (decode / verify passes <= 7 tokens), the
xe tier's TP hooks (residency table + version + dirty keys, CPU pool share, promotion between passes), dense state
pointers / write tracking / per-slot Engram reads, rank 0 reading the engine's planar weights in place (-1.1 GB on
card 0).  Phase A bench (synthetic state, all experts VRAM): N = 1/2/3/4 decode 23.1 / 16.9 / 14.8 / 13.3 ms a pass,
4-token passes 43.7 / 30.4 / 27.7 / 23.3.

In the engine (essay x400, bias 0.5 / 0.3, clocks pinned, STRATA_XE_FLEX_GIB=3 on both sides): plain decode 45.0 /
45.5 -> 60.5 / 60.6 tok/s (+34 %); DSpark 59.95 / 59.11 / 60.06 -> 61.53 / 60.62 / 65.66 (+5 %), verify 33.3 -> 29.3
ms/pass (-12 %), but the draft 5.1 -> 6.2 ms/pass (not explained yet).  (This A/B ran on the build just before rank
0's in-place weights - 1.1 GB more of card 0 for experts since, not re-timed; the PPL / 95K runs used the final build.)  A plain pass at steady state (STRATA_TP_PROF):
15.0 ms = the cards 11.9 (the host's waits on the mailboxes), CPU-pool experts 1.3, Engram 1.0, the rest 0.8; a
verify pass 26-30 ms = the cards 15.7-17.5, CPU-pool experts 7.2-8.9 (10.7-11.3 layers a pass have one).  Decode-path
PPL (p2k, exact routing): today 2.0142 / 2.0243, TP 2.0051 / 2.0059; KL today-today 0.0068, TP-TP 0.0030, today-TP
0.015-0.021 (top-1 97.9-98.3 %).  95K prefix (ctx 131072): prompt 1408 / 1447 tok/s, DSpark 48.6 / 49.2 (neutral).

The service (ctx 262144, STRATA_TP=4, FLEX 3): the first long prompt failed ("tier run refused", then
UR_RESULT_ERROR_OUT_OF_RESOURCES) - TP's weights (~2 GB a card, ranks 1-3 2.75-2.87 GiB with the KV replicas: ~0.85 GB
a card at 262K) and the bigger flex left the prompt path's buffers no room.  Reverted (STRATA_TP=0, FLEX 1).  The
limiter in every run: CPU-pool experts (TP 1.4-2.6K vs 1.1-1.6K uses a run): TP's VRAM displaces ~600 experts.

Lessons (measured): host -> GPU flags need fence + volatile polls (atomic acquire loads miss the host's writes:
bench/hs_probe2); a second in-order queue DOES run beside a spinning one (bench/cq_probe: 16 MB in ~710 us); the first
card-to-card copy of each pair sets up the path (~6 ms; init does it before any cross-card wait); stacked Q6_K weights
must be ONE plane set; the promotion cache needs several jobs a card in flight (its pc_setup may run first on the
tier's own path: DSpark's 16-token seed passes); graph recording ~40 ms a width - pre-recorded at attach, cap floor
1024 rows.

## Status 2026-10-23 ~02:00 (at the service's settings)

The FLEX 3 failure above was the expert settings, not TP: FLEX 3 GiB with a 0.3 GiB margin left ~0.3 GiB free on the
expert cards, and a 252-token prompt (below the prefetch path, which resizes the flex region) ran out of VRAM in
run_big.  At the service's own settings (ctx 262144, FLEX 1 / margin 1.5, RESERVE 3.5) eager TP was DSpark -2 %,
plain +15 %: its KV replicas displaced ~960 VRAM experts (CPU-pool uses 2x), and the four normal-priority promotion
helpers (host repacks, 10-20 ms an expert) slowed the draft's submission (enqueue 3.95 vs 2.30 ms) and the verify
pass (one helper: 34.1 -> 30.7 ms).  Fixed:

* STRATA_TP_LAZY_KV: ranks >= 1 keep the KV replicas in Level Zero virtual memory (one address for the graphs),
  pages mapped as a pass's cap bucket or a sync reaches them, the new rows copied from card 0.  The VRAM not mapped
  yet is lent to the tier's flex region (placement as with eager replicas); tier.tp_make_room takes back only the
  shortfall when rows map (keeping 1.5 GiB free there emptied the flex after a long prompt: no promotions, 95K
  DSpark 43.0 vs today 48.5).
* STRATA_TP_PC_DEVREPACK: promotions copy the file layout (pinned arena blobs straight from the arena) and the card
  repacks (slot -> temp -> slot); 2 helpers.  Checked: 0 bad slots.
* The head split (the draft's LM head, small passes' Engram projections) reads TP ranks' row slices (same rows,
  same planar layout) instead of copies - it now fits after long prompts.

* Both paths: the expert cards' last 1 GiB slab is filled at placement (STRATA_XE_FILL_SLAB) - its tail was
  allocated anyway (today's path +0.74 GiB of experts a card, CPU-pool uses -10 %).

Measured (one-shot, service settings, clocks pinned, both paths with the slab fill): essay DSpark today 60.5 / 56.9
tok/s, TP 61.2 / 64.7 (+7 %; verify 33.5 -> 28.8 ms); 20K prefix 92.0 -> 110.0; 95K prefix 50.4 -> 50.3 (verify
40.5 -> 38.7 ms, fewer tokens a pass); plain 47.1 -> 61.6 (+31 %); PPL p2k 2.0085, KL vs eager TP 0.0033.  Free
VRAM is logged per card (load, warm-up, graphs, each profile window).

Next: the draft's remaining +0.55 ms under TP; card 0's 3/4 of the sharded dense weights (~3.8 GB, only the prompt
path reads them: page them in per layer from the other cards); the CPU pool's share of verify passes (GPU help over
PCIe for pinned arena experts); the pass's own kernels (unitrace of tp_bench); then TP prefill and the draft on the
TP core.

## Status 2026-10-23 ~05:30 (the pager: card 0's copies of the sharded weights)

With TP on, only the prompt path (layer-major batches) reads card 0's full copies of q_b, wo_a, wo_b, the shared
expert (gate | up, down) and the Engram projection: 4.93 GiB that the ranks hold in their shards already.
STRATA_TP_PAGE=1 (tp/tp_pager.cpp) pages them:

* The dense half allocates them through `Ds41DenseConfig::paged_alloc`: one unit of Level Zero virtual memory a layer,
  a ggml buffer over the region (`ggml_backend_sycl_buffer_from_ptr`), every page mapped while the weights load and
  reorder.
* After TP init, `enable()` builds each tensor's fill plan from the ranks' shard layouts (`Ds41Tp::shard_desc`: rows,
  gate | up halves, K blocks, slices), gives the load-time pages back and keeps 3 slots (244 MiB) plus 32 MiB of
  staging.  It waits until free VRAM shows the driver's lazy free; the tier's dense cap takes the freed bytes
  (`tp_dense_bonus`): card 0 holds 1690 experts instead of 1326 (one-shot); the service's deferred cache opens with
  14.2-16.9 GiB of experts instead of 10.0-11.3.
* `need(l)` (the dense half's attn_router spans, before the submit lock) maps layer l's unit into a slot and fills it:
  card 0 pulls rank k's shard (card-to-card copies on its own copy queue) straight into the unit for row, half and
  slice splits; K splits through the staging and a scatter kernel on the dense queue (2-D card-to-card copies ran at
  31-40 MB/s); the dense queue waits for them.  The next layer's copies start meanwhile.  `touch(l)` after the layer's
  work is queued: the slot is free once that work is done.  The copies are never the other cards' (see below).
* Rank 0 owns its copies of the paged tensors (no shared planes for them).  Prompt passes of <= 7 tokens go to TP
  passes, and DSpark seeds in chunks of 7: the dense half no longer has the weights for a small pass at hand.

Checked: STRATA_TP_PAGE_CHECK=1 refills every unit in place and compares it with the load (bit-identical).  Measured
(one-shot, service settings, clocks pinned): essay DSpark 69.6 / 69.0 / 71.1 tok/s (TP without the pager 63.0), 95K
prefix DSpark 57.0 (50.3), prompts +4 %, plain 63.5; no slot waits.

**The first service deploy (05:20) garbled long prompts.**  tsvc2: the 8K / 16K / 31K needles came back as word salad
(the decode ran to its length limit), one 8K in two passed, and the tool call after them was garbage too; the
one-shot runs (one prompt each) had been fine.  The fills were then the OTHER cards' copies (rank k's queue writing
into card 0's virtual memory).  A private-port copy of the service (bench/engine1/tpg.sh: the tsvc2 sequence with the
answers shown) reproduced it exactly; the pager off with an 18 GiB cache cap passed (not VRAM pressure), and so did
the pager with its copies on card 0's queue (STRATA_TP_PAGE_ON0) or with no remapping at all (STRATA_TP_PAGE_KEEP).
bench/engine1/vmm_remap.cpp has it in 30 lines: after card 0 remaps an address to new pages, card 1's copy into it
still lands in the OLD pages (every time; card 0's own copy into it is right, and a cross-card barrier waits
correctly).  A slot moving to another layer left the other cards writing into the pages of whichever unit or freed
load-time pages had held that address - other layers' weights, then the expert cache that took the freed VRAM.  Fix:
card 0 pulls every fill on a copy queue of its own.  Checked: the refill check bit-identical through it, the service
sequence twice in one server all right (needles, tool call, svcbench text), prompts as without the pager (8K
764-953 tok/s, 16K 1310-1429, 31K 1670-1721; pager off 784-928 / 1345 / 1690); short prompts (<= 34 tokens) run as
TP passes, 55-83 tok/s against 30-40.  engine1's lazy KV replicas are safe: their pages are only ever added, and
each card copies into its own.

In the service since 06:02 (tsvc2 against the last run without the pager): needles 200 / 8K / 8K / 16K / 31K all
found, the tool call right; essay 69.5 / 66.0 tok/s (65.1 / 63.4), svcbench 80.3 / 76.2 (80.4 / 81.6: its noise),
prompts 8K 795 / 927 tok/s (786 / 942), 16K 1162 then 1419 / 1437 (1342), 31K 1646 (1689); hidden RAM 40 GiB peak.

Next: the CPU-pool share of verify passes (pinned arena experts computed by idle cards over PCIe), the draft's
remaining +0.5 ms under TP, a bit-exact prompt path if a test ever needs repeatable rows, TP prefill.

**Row 111 (not the pager).**  In p2k PPL runs (one 2048-token batch, exact routing) row 111 - token 1018 in a repeated
pattern, the next token (6201) certain - predicts 30 (once 201) in some runs: NLL 6.8-7.0 instead of 0.004, PPL
+0.006.  First seen with the pager (2 of 3 runs), then in every variant tried: pages kept, no prefetch, fp16 cache
off, refills on card 0's queue, a 3 ms busy kernel on the dense queue with no pager at all; and with the pager off
(bench/engine1/ttp18.sh: 2 of 8).  In all: 2 of 14 runs with the pager off, 9 of 27 with it or a variant - the
same rate within the noise of such counts.  STRATA_DUMP_COL=<column> writes per layer that column's routing, FFN
input and routed sum (logs/dump_col.<pid>.bin).  On the bad runs (one with the pager, one without):

* the routed sums match the F32 reference experts from the GGUF in every layer (bench/engine1/test_ds41_dumpref.cpp:
  rel 1e-7..1e-2) and the routing matches the router recomputed from the dumped input (bench/engine1/routercheck.py:
  the same top-6 in all 40 layers, weights within 2e-6).  The huge outputs from layer ~21 on (|routed| up to 43,000)
  are the model's own response to that input;
* good runs' FFN inputs for that row already differ 2.5e-4 at layer 1, ~30 % by layer 19 and up to ~80 % later, with
  the same logits (KL <= 0.001) - the pager off as much as on; the bad runs leave that spread smoothly at layers
  19-21 for another attractor.  Nothing jumps.

So the row is chaotic and the engine's run-to-run noise decides it.  Noise sources seen (every config): the placement
varies with free VRAM at load (8272-8275 VRAM experts) and so do the prompt prefetch buffers (0.72-0.74 GiB a card),
which moves experts between the resident (planar, int8 activations) and the streamed kernels; and the dense half
differs ~1e-7 at layer 0 already.  Row 0, which attends only to itself, differs 1.7e-7 at layer 0 and 3.5e-5 at
layer 1: wherever activations are rounded to 8 bits (dp4a GEMMs, planar experts) a last-bit difference flips a
rounding, and the jump is ~1e-5.  The bulk statistics do not move with the pager: KL against a resident run
0.0021-0.0025 mean, PPL without row 111 1.998-2.009.  A bit-exact prompt path (fixed reduction orders, placement
independent of the driver's free-VRAM reading) would make such rows repeatable, not right; not planned.

## Status 2026-10-23 ~17:30 (the CPU-pool experts of verify passes: promotions in time for the next pass)

Measured first, on the DSpark essay at the service's settings (`STRATA_XE_CPU_LOG=<file>`: a line per TP pass, per
CPU-pool expert, per layer with any and per promotion event; `bench/engine1/cpulog.py` reads it, `pcsim.py` replays
a run's expert uses under other promotion policies):

* 94-95 % of the CPU pool's experts are pageable arena experts.  The pinned ~20 GiB of the arena hold the hottest arena
  experts, and card 0's cache takes ~1,700 of those, so the plan (idle cards computing pinned experts over PCIe) would
  have covered ~5 % of the CPU pool's work.
* The pool reads 28-31 GB/s cold (larry's DRAM 38-40) and 45 GB/s with the weights in the caches
  (`STRATA_XE_CPU_TWICE=1` runs each layer's CPU experts again): faster kernels could give ~1.3x at most.
* 63 % of the CPU uses were repeats, 31 % of an expert the pass before had run on the CPU.  A promotion took 2-5
  passes: the bounce was filled after the pass (helpers), the copies went out 64 chunks of 1 MB a pass, and the slot
  was published one pass after its copies.  18 % of the CPU uses came while that expert's promotion was in flight, and
  the in-flight cap (8 a card) skipped 533 promotions.

So promotions now land before the next pass:

* `STRATA_TP_PC_EARLY` (default 1): a pageable expert the next tp_between_passes will promote gets its pinned bounce
  filled right after the CPU pool ran it, while the pass goes on, with streaming stores (`STRATA_XE_PC_STREAM`: no
  read-for-ownership of the bounce, and its lines kept out of the caches the pool uses).  Between passes it only gets
  its card and slot; a bounce no card took (every card at its cap) goes back once filled.
* Copy chunks of 4 MB under TP (were 1 MB; 64 a pass are ~20 experts): they run on the cards' promotion queues during
  the draft, when cards 1-3 are idle.
* `STRATA_TP_PC_PUBLISH` (default 1): the TP core calls the tier's dirty hook before every pass (it used to only when
  the residency table's version had changed), and the tier publishes there the promotions whose copies are done.
* `STRATA_TP_ENG_ASYNC` (default 1): a pass's Engram rows are read by a helper thread, so CPU-pool layers among 0-3 (a
  quarter of the verify passes have one) no longer wait behind both Engram layers' rows (~2 ms).  Neutral within the
  noise below.

A DSpark cycle (verify pass + draft) by pass width, from the pass timestamps the CPU log carries, weighted by one
pass-width mix for every run (the acceptance moves with the text, and the texts diverge after a few tokens):

| one-shot, service settings, essay 1000 tokens, clocks pinned | CPU-pool experts a pass | one-expert CPU layer | cycle |
| --- | --- | --- | --- |
| before (2 runs) | 10.2 / 11.3 | 391 / 413 us | 31.5 / 32.6 ms |
| promoted at the first use (`STRATA_XE_PROMOTE_MIN=1`, 2 runs) | 7.9 / 8.1 | 473 / 480 us | 30.7 / 30.9 ms |
| promoted at the second use, Engram inline (3 runs) | 7.0 / 8.0 / 8.0 | 453 - 481 us | 29.9 / 30.1 / 30.5 ms |
| the same with the Engram rows on the helper (2 runs; in the service) | 7.2 / 8.0 | 454 / 455 us | 30.1 / 30.2 ms |

A cycle is 6 % shorter, ~+6.5 % tokens a second at equal acceptance.  Promoting at the first use evicts more and
slows the draft (the copies of ~7 promotions a pass run during it).  Pausing the fills while the CPU pool computes
(`STRATA_TP_PC_PACE=1`) brought one-expert layers to ~440 us, but later fills missed more next passes: cycles 30.2 /
30.5 ms.  Checked: `STRATA_XE_PROMOTE_CHECK=1`, 1392 promotions, 0 bad slots; a private-port copy of the service
(`tpg.sh`, the tsvc2 sequence twice) and tsvc2 on the service (17:31): needles 200 / 8K / 8K / 16K / 31K found, the
tool call right, essay 69.9 / 63.6 tok/s, svcbench 80.4 / 72.9 (the service samples at temperature 1: its acceptance
noise is larger than 6 %), prompts 8K 816 / 932, 16K 1303, 31K 1673 tok/s, hidden RAM 40 GiB peak.  The one-shot texts
diverge from the old path's after a few tokens (promoted experts run the cards' int8 kernels, not the CPU pool's) and
read as well.

What is left of the CPU pool's share: ~40 % of its experts are an expert's first use in the run (no cache takes those:
only a faster CPU part or routing prediction); more slots (card 0 has ~2 GiB free in decode) would take ~6 % more of
the rest at MIN 2 (`pcsim.py`).  The service drops the promotion cache at every batch prompt (the flex regions become
the prompt's prefetch buffers), so its sessions start cold more often than these runs.

## Status 2026-10-23 ~20:35 (the verify pass's own kernels, and where the rest of its time goes)

Where a pass's GPU time goes, measured two ways: `tp_bench` eager under unitrace (each kernel's device time) and
`STRATA_TP_STAMP_LAYER=l` - a kernel reading the sub-group clock after every step of layer l on every card, recorded
into the pass graph (~0.6 us a stamp; the device-scope clock is not supported on the B70, the sub-group one ticks
~2790 a us with the clocks pinned and agrees across kernels); `-1` stamps every layer.  The cards are busy all the
pass: the kernel sum per card-layer equals the pass time (325 us at 1 token, 556 at 5).  At 5 tokens a card-layer
(bench, layer 20): Q6_K mat-vecs ~165 us (the dp4a kernels slow 30-60 % at 4-6 columns), routed experts 122-193,
attention 55, the indexer 84 and the compressor 45 (layer 20 has both), the two all-reduce copies 16.5 us each (PCIe:
~26 GB/s out of a card - `STRATA_TP_PUSH`'s kernel stores gain nothing), the two hc steps 15 each, the router mat-vec 21.

Done (all bit-identical unless noted):

* `route_jobs` runs a lane a pick (a work-group: dedup + scan in local memory) - the one-work-item form's dependent
  loads were 4.5 us at 1 token, 15 at 5 (a pass -0.08 .. -0.44 ms by width).
* The expert kernels stop at route_jobs' real job count (in `need_cpu[1]`) and gate/up is launched for 8 job slots,
  each work-group looping over the jobs (`STRATA_TP_NJ_DEV`, `STRATA_TP_JOB_GRID`; the null slots of n x 6 were
  dispatched and walked): a 5-token pass -1 %.
* The Q6_K mat-vecs take a form per shape and width (`STRATA_TP_MV_TUNE`): fastmv's dispatch, the dp4a kernel's row
  blockings (2 / 4 rows, normal / 256 registers) or the XMX kernel with a K split, timed at init on the card's own
  40 layers (cold weights) in replayed graphs with a small kernel between the launches - the XMX kernels start ~5 us
  slower after another kernel than after themselves (`bench/fm3/tpmv_bench`), which the first, back-to-back tuning
  missed.  An XMX form must beat dp4a by its producer's extra cost: the producers (hc step, q_a norm, attention,
  quant_xT, SwiGLU) write the q8 a second time in the XMX layout when the consumer takes it (the hc step's scattered
  2-byte stores cost ~3 us, attention's ~2).  At 5 tokens that is XMX for q_b (32.7 -> 24.6 us), wo_b (31.5 -> 24.7)
  and the shared expert's down (10.3 -> 8.6), dp4a's 4-row form for q_a+kv / wo_a / gate|up.
* The XMX mat-vec is exact: B = q - 32 and the scale applied after the MAD (a scaled B reaches 4064, past fp16's
  exact integers, and moved the decode-path PPL ~+0.005); slightly faster too.

Measured (one-shot, service settings, clocks pinned): `tp_bench` 5 tokens 22.96 -> 22.06 ms a pass (-3.9 %), 3 tokens
-0.8 %; the DSpark essay's cycle at the common width mix 29.70 / 30.17 ms against 30.39 (the mean of five runs of
today's code, 30.03-31.15).  Decode-path PPL in 5-token passes (exact routing): the tuned forms' KL to fastmv's
dispatch 0.0029 against the run-to-run floor 0.0027; PPL without the chaotic row 111 2.0073 / 2.0087 vs 2.0059 /
2.0037.  No gain: the down kernel's jobs split over its sub-groups (`STRATA_XE_DOWN_SPLIT`, +1.9 %: a work-group
waits for its slowest job), XMX for q_a+kv and wo_a below 6 tokens (their producers' cost).  Also settled: the
service's 64 promotion chunks a pass cycle as the A/B's 256 (30.05 / 31.15 vs 30.03 / 30.13 / 30.61 ms).

**Where the rest goes: the cards waiting for each other.**  Stamps on every layer of the essay (`tr5_newst`): every
card waits ~4 ms of a ~24 ms pass at the FFN all-reduce, and card 0 waits 2.2-2.8 ms for the CPU pool; code waits
~6 ms (CPU 4.7), math ~8.7 (CPU 7.8: its routing leaves VRAM most).  Two patterns: in layers 11-19 and 32 the CPU pool
has the most experts and every card waits for card 0, which waits for the host; in layers 25-39 card 0 holds almost
none of the hot experts (11-36 us of expert work against 80-200 on the others) and idles - its cache is the next
slice of the global ranking after cards 1-3's.  The static part (per layer, the slowest card's average lead) is
~2.2-2.4 ms a pass on the essay, but a placement re-dealt from one text's usage counts (`STRATA_XE_USAGE_OUT`,
`bench/engine1/place.py`) does not transfer: built from the essay it loses 1.5 ms on code, from math 2.9 ms on code;
from code it gains +0.3-0.5 ms on the others.  The experts run near the memory roofline already (fitted: 837 GB/s +
19 us a job), so the levers are dynamic: replicas of each layer's hottest experts (the top 8 carry 22-29 % of a
layer's jobs) on card 0's free decode VRAM (~3 GB), handed out per pass by route_jobs to the least-loaded holder
(card 0 starting at its CPU estimate); and the CPU pool's latency (~455 us for a one-expert layer, ~370 of it the
expert's bytes at 30 GB/s).

