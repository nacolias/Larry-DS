# engine1 - DeepSeek-V4.1 decode / verify on N cards (tensor parallel), host out of the loop

Status: phases A-C built and measured, in the engine behind STRATA_TP=N (default 0); not in the service yet (VRAM: see
the status at the end).  User go-ahead 2026-10-22 ("option 3": the four-card engine, TP degree a setting, any card
count).  Mirror of larry:/opt/ai/dsv41/xolotl/tools/v41/engine/tp/ (this file is DESIGN.md there).

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

Next: VRAM back to the experts and the prompt path - KV replicas sized to the context reached (grown on demand), the
TP weights inside the tier's budget accounting so a service-sized flex fits; then the pass's own kernels (unitrace of
tp_bench, eager; STRATA_TP_PUSH untested), the CPU pool's share in verify passes, then TP prefill (drops card 0's full
dense copy: 6.5 GB) and the DSpark draft on the TP core.
