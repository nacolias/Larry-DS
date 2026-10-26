#!/bin/bash
# ds41-xe-engine.sh - the DeepSeek-V4.1 engine with the Intel Arc tier, as Strata's server starts it
# (`<exe> --serve <args>`): larry's oneAPI / Level Zero environment, the serve-path defaults of upstream's
# tools/v41/serve/ds41-engine.sh (DSpark speculative decoding, exact prompt routing), and the Xe tier's knobs.
ARGS=("$@")
# Larry-specific (a Ryzen 7 5800X with 121 GiB of RAM, four 32 GB B70s on one x16 switch): STRATA_HOST_CPUS,
# STRATA_XE_POOL_WORKERS and STRATA_XE_HELPER_CPUS follow its core layout (CPU n and n + 8 are one core);
# STRATA_XE_PINNED_GIB stays under the xe driver's TTM limit (all GPU-driver host memory ~60 GiB there); the VRAM
# budgets assume 32 GB cards.  Every setting is a default: a variable already set in the environment wins.
# References below: README = larry/ENGINEERING-LOG.md, engine1/DESIGN.md = tools/v41/engine/tp/DESIGN.md,
# bench/... and patches/... = larry/bench/..., larry/patches/..., abwT<n> = a numbered A/B window in the log.
source "$(dirname "${BASH_SOURCE[0]}")/xe-env.sh"
# DSpark (the checkpoint's own mtp.0-2 draft stack, Q2_K sidecar): exact speculative sampling; '' = off
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-$SIDECAR}"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
# the draft confidence that ends a pass's drafts: 0.3 (was 0.6) since the draft is 4.5 ms - wider passes pay
# (one-shot essay 54.3 -> 58.4 tok/s, p1 73.0 -> 75.3; 0.1 / 0.2 / 0.45 / 0.75 lower)
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.3}"
export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-0}"
# the draft stack's Q8_0 projections requantized to Q6_K at load: fastmv kernels (wo_a one launch, not 8); draft 7.7 -> 6.5 ms/pass
export STRATA_DSPARK_DENSE_TYPE="${STRATA_DSPARK_DENSE_TYPE-q6_K}"
# prompt tokens route exactly; turns <= 96 new tokens run as exact passes
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# the Xe tier: every expert not on a card goes to the RAM arena (pinned up to STRATA_XE_PINNED_GIB = 24: TTM caps all
# GPU-driver host memory at ~60 GiB on larry; the rest is pageable); MemAvailable keeps a 16 GiB floor at load
# (24, was 16: the tier's own VRAM no longer costs host RAM - service "hidden" peak 32 GiB at 16; 8.5K prompt +4% at 28)
export STRATA_XE_PINNED_GIB="${STRATA_XE_PINNED_GIB:-24}"
export STRATA_XE_MEM_FLOOR_GIB="${STRATA_XE_MEM_FLOOR_GIB:-16}"
# the dense card's expert share: 14 GiB (decode DSpark +7.5% vs 10: fewer CPU-pool experts), 1.75 GiB kept free (was 4:
# the replicas below took ~2.5 of it; with the 1M context's KV state the cache is worth more than they are - 1M agent
# replay prompts 103.3 -> 99.0 s, decode 112.6 -> 115.6 tok/s with 0.25 GiB of replicas, 2026-10-24 22:20)
export STRATA_XE_DENSE_GIB="${STRATA_XE_DENSE_GIB:-14}"
export STRATA_XE_DENSE_RESERVE_GIB="${STRATA_XE_DENSE_RESERVE_GIB:-1.75}"
export STRATA_XE_RESERVE_GIB="${STRATA_XE_RESERVE_GIB:-3.5}"   # per expert card: room for the prompt path's buffers (8192-token batches)
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"   # dense Q8_0 -> Q6_K at load (upstream default; -23% bytes)
export STRATA_ENGRAM_REQUANT="${STRATA_ENGRAM_REQUANT-q6_K}"   # the two Engram KV projections F16 -> Q6_K (315 -> 129 MB a token; PPL within noise)
export STRATA_ENGRAM_THREADS="${STRATA_ENGRAM_THREADS-256}"   # the Engram row reader pool = queue depth over both NVMe drives (32: a cold 31.6K prompt waited 1.8 s for batch 1 rows, 256: 0.4 s)
# verify passes: no D3 routing prediction (a graph + a host readback per layer per pass; the Xe tier has no prefetch)
# the serve batch-width fit (Xe: the chunk buffers live on the expert cards; ~0.32 MiB of state a column on card 0)
export STRATA_SERVE_BATCH_FIXED_GIB="${STRATA_SERVE_BATCH_FIXED_GIB:-3.5}"
export STRATA_SERVE_BATCH_COL_MIB="${STRATA_SERVE_BATCH_COL_MIB:-0.45}"
export STRATA_D3_PF="${STRATA_D3_PF:-0}"
# the lightning indexer of decode / verify passes (<= 8 queries) on XMX: the per-row kernel took ~23 ns a key row a
# query (98K context ~2.3 ms an indexer layer per query, 8 indexer layers); a 95K-context DSpark essay 21.6 -> 53-55
# tok/s with both (short context and decode-path PPL unchanged)
export GGML_SYCL_LI_GEMM_MIN="${GGML_SYCL_LI_GEMM_MIN:-1}"
export GGML_SYCL_LI_XMX="${GGML_SYCL_LI_XMX:-1}"
# the prompt indexer (512-query spans) as one fused XMX kernel: li_gemm wrote the [B*32 x cap] f32 score matrix and
# read it back (5.7 ms a call at a 31K prompt's second batch, ~10 % of card 0) - fused 16K keys 4.36 -> 0.63 ms,
# 64K 17.4 -> 2.33 ms (bench/li/li_probe); and told each query's visible rows instead of a [cap, n] 0 / -inf mask
# (its repeat / add / step / log / F16 copy, ~0.9 ms a call) - abwT112: fresh 16K 7.00 -> 6.76 s, 31K 12.34 -> 11.31 s,
# the agent replay's follow-ups 42.6 -> 39.9 s
export GGML_SYCL_LI_FUSED="${GGML_SYCL_LI_FUSED:-1}"
export STRATA_LI_NVIS="${STRATA_LI_NVIS:-1}"
# card 0's 512-column prompt span: wo_a's 8 slices as one product written in wo_b's layout (768 us + a 162 us copy ->
# 520 us) and the hc mixes' rms_norm copy + [24 x 20480] GEMM as one pass over the state (2 x 323 -> 2 x 153 us)
# - abwT123: fresh 16K 6.74 -> 6.40 s, 31K 11.51 -> 10.96 s (first runs 11.71 -> 10.78); the agent replay's side
# prompts 16.09 -> 15.30 s
export STRATA_OPROJ_A="${STRATA_OPROJ_A:-1}"
export STRATA_HC_NORM_MV_WIDE="${STRATA_HC_NORM_MV_WIDE:-1}"
# the fused hc ops at prompt widths too (patch-hcwide.py's wide hc_fa: 16 tokens a chunk's work-group, the mixes' rows
# staged once): the separate hc_post / hc_pre / mixes and their state copies go - abwT124: fresh 16K 6.45 -> 6.23 s,
# 31K 10.90 / 10.96 -> 10.55 / 10.35 s (here the prompt path's STRATA_HC_NORM_MV_WIDE is then unused)
export STRATA_HC_FUSE_MAXN="${STRATA_HC_FUSE_MAXN:-16384}"
# the prompt attention's output in fp16 straight into wo_a's op (with STRATA_OPROJ_A): no f32 copy of it, no
# conversion back
# - abwT129 (two servers each): fresh 16K 6.25 / 6.29 -> 6.17 / 6.16 s, 31K 10.32-10.35 -> 10.16-10.27 s;
# card 0 ~0.26 ms a span lighter (wo_a 543 -> 337 us), the rest now waits on the expert cards
export STRATA_SDPA_O16="${STRATA_SDPA_O16:-1}"
# a decode / verify job's own token count picks its expert kernel form (1-token jobs of verify passes 9-17 % faster)
export STRATA_XE_JOBNT="${STRATA_XE_JOBNT:-1}"
# prompt chunks: each expert card pushes its part to the dense card when it is done (31.6K prompt 23.0 -> 21.5 s, exact)
# prompt chunks: the expert cards push their parts to card 0 as fp16 (2; 1 = f32): half the bytes over card 0's link -
# 16K 10.54 -> 10.17 s, 31K 17.69 -> 17.41 s (2026-10-25 abwT08)
export STRATA_XE_COMBINE="${STRATA_XE_COMBINE:-2}"
# prompt calls without the prefetch (< 2048 tokens): an arena expert the call would stream that serves <= 4 tokens runs
# on the CPU pool beside the cards (a pageable one streamed is read + written by the bounce copy and read by the DMA;
# the pool reads it once).  Service A/B (DSpark, clocks pinned, 5 runs): pp512 166 -> 231 tok/s, at 8K 159 -> 222, at
# 32K 154 -> 219, pp160 91 -> 125, 2047 tokens ~390 -> ~490; long prompts unchanged; prompt-path KL at the run-to-run
# floor (bench/speed/tab_all.sh, tlppl.sh).  0 = every arena expert streamed
# 3 with BAR staging (was 4): the light experts' streams cost less than the pool's decode of them now - fresh 1K-4.5K
# prompts -0.05..-0.1 s, 512 / 16K / 31K even (abwT87, abwT88)
export STRATA_XE_CPU_LIGHT="${STRATA_XE_CPU_LIGHT-3}"
# ... but not the pinned arena's light experts: they reach the cards by DMA from where they are (one DRAM read, no CPU),
# so the pool takes only pageable ones (a pageable expert streamed costs ~3 DRAM reads: copy into a ring + the DMA) -
# fresh 512 / 1K / 2K / 3K prompts 2.19 / 2.70 / 3.67 / 3.92 -> 2.08 / 2.43 / 3.33 / 3.79 s, 4.5K-16K the same or a
# little faster (abwT69, abwT70).  1 = the pool takes pinned ones too
export STRATA_XE_CPU_LIGHT_PINNED="${STRATA_XE_CPU_LIGHT_PINNED-0}"
# the prompt prefetch (every arena expert of a layer staged ahead) from 4096 tokens a batch, not 2048: a whole prompt of
# 2-3K tokens runs faster on demand with the CPU pool (2,047 tokens 436 -> 502 tok/s, 2,900 455 -> 588); a long
# prompt's upper layers (enc-skip: its last ~2,560 tokens over the whole context) keep it from 2048 - without that a
# 31K prompt took 5 s longer (23.0 / 23.4 s with it)
# 3072 since 2026-10-26 (was 4096): a chat's 3.6-3.8K-token follow-ups at depth -0.65..-0.87 s (their layers 0-19 no
# longer stream on demand); 2048 made 1.7-2.2K ones 0.1-0.45 s slower (abwT103)
export STRATA_XE_PREFETCH_MIN="${STRATA_XE_PREFETCH_MIN-3072}"
export STRATA_XE_PREFETCH_MIN_B="${STRATA_XE_PREFETCH_MIN_B-2048}"
# q_a + kv (both read the attention norm) as one weight: one mat-vec / GEMM, not two (DSpark verify -3.6 %, 31.6K prompt 22.4 -> 21.5 s; PPL unchanged)
export STRATA_QKV_FUSE="${STRATA_QKV_FUSE:-1}"
# larry, engine1: decode / verify passes (<= 7 tokens) on the four cards' tensor-parallel core - the dense half split by
# heads / columns over the cards, each card its own experts, a replayed SYCL graph a card a pass; its KV replicas map
# as the context grows (the flex region holds that VRAM meanwhile), promotions repack on the cards.  One-shot A/B at
# these settings (ctx 262144, FLEX 1): DSpark essay 58.7 -> 63.0 tok/s, 20K prefix 92 -> 110, 95K 50.4 / 50.3, plain
# 47 -> 62 (engine1/DESIGN.md).  0 = the old path
export STRATA_TP="${STRATA_TP-4}"
export STRATA_TP_CACHE="${STRATA_TP_CACHE-$SRV/tpcache}"
# engine1's pager: card 0's copies of the dense weights the TP cards hold (q_b, wo_a, wo_b, shared expert, Engram
# projection: 4.9 GiB) paged in per layer for prompts only; card 0 takes ~360 more experts (DSpark +10 %, 95K +13 %
# one-shot).  The p2k row that sometimes goes wrong (row 111) does so with the pager off too - a chaotic row, not a
# race.  The first deploy (05:20) garbled long prompts: the other cards wrote the fills into card 0's remapped
# virtual memory, which lands in its old pages (bench/engine1/vmm_remap.cpp); card 0 pulls them now (2026-10-23,
# engine1/DESIGN.md).  0 = card 0 keeps the copies
export STRATA_TP_PAGE="${STRATA_TP_PAGE-1}"
# engine1's replicas: card 0 copies each layer's hottest expert-card experts (the uses of three texts in
# replica-usage.txt; 2.5 GiB pinned on the host, uploaded at every seed where card 0 keeps 1 GiB free) and route_jobs
# hands a pass's job of one to whichever holder finishes sooner.  One-shot A/B: essay cycle -2 %, math -6 %, code verify
# -3 %.  A full card takes them back before an allocation fails (ggml-sycl's reclaim hook: the first try crashed at
# 21:41 on a 1.9 GiB pool growth after the replicas had the room; engine1/DESIGN.md).  0 = none
# (0.25 since the 1M context: card 0's VRAM goes to its expert cache first - see STRATA_XE_DENSE_RESERVE_GIB)
export STRATA_XE_REPLICA_GIB="${STRATA_XE_REPLICA_GIB-0.25}"
export STRATA_XE_REPLICA_USAGE="${STRATA_XE_REPLICA_USAGE-$D/replica-usage.txt}"
# engine1's draft: DSpark's mtp.0-2 stack on the four cards too (its blocks split as the trunk's layers, its 128 Q2_K
# experts a block over the cards; ~1.2 GiB a card instead of 5 GB on card 0, whose expert cache takes the difference):
# a draft step 4.9 -> ~3 ms, one-shot DSpark cycle essay -7 %, code -9 %, math ~-3 % (noisy).  Promotions write the
# planar bytes on the host and the cards only copy them (no kernel beside the draft).  0 = the ggml draft on card 0
export STRATA_TP_DRAFT="${STRATA_TP_DRAFT-1}"
# larry's host CPUs (a 5800X: CPU n and n + 8 one core): the CPU pool's 6 workers on cores 1-6, the TP pass's host loop
# (its mailbox spin and its share of the pool's work) on CPU 0, the promotion fill and Engram helpers on core 7 (CPUs 7,
# 15).  Floating, the host loop and the helpers landed on the workers' SMT siblings: the pool's one-expert layers
# 543 -> 481 us (math), 523 -> 452 (essay), the DSpark cycle -1.6 / -2.6 % (engine1/DESIGN.md).  Unset = floating
export STRATA_HOST_CPUS="${STRATA_HOST_CPUS-0}"
export STRATA_XE_POOL_WORKERS="${STRATA_XE_POOL_WORKERS-6}"
export STRATA_XE_HELPER_CPUS="${STRATA_XE_HELPER_CPUS-7,15}"
[ "$STRATA_TP" = "0" ] && unset STRATA_TP
# the promotion cache (each expert card's flex region, also a prompt's prefetch buffers): 1 GiB a card, 1.5 GiB kept
# free (3 GiB / 0.3 left a 252-token prompt's chunk buffers no VRAM: OUT_OF_DEVICE_MEMORY, 2026-10-23)
export STRATA_XE_FLEX_GIB="${STRATA_XE_FLEX_GIB-1}"
export STRATA_XE_FLEX_MARGIN_GIB="${STRATA_XE_FLEX_MARGIN_GIB-1.5}"
# a prompt batch that ends before context position 5000 takes no prefetch: its routed arena experts stream on demand
# and the CPU pool runs the light ones (CPU_LIGHT).  (8192 until 2026-10-25: with the dense card out of the tier's
# calls below, 6-8K batches from empty gain from the prefetch - 8K 7.24 / 7.02 -> 6.86 / 6.71 s - and 4K ones do not,
# abwT18 / abwT19)
export STRATA_XE_PREFETCH_DEPTH="${STRATA_XE_PREFETCH_DEPTH-5000}"
# lookup drafts (STRATA_SERVE_LOOKUP) stay within the TP core's 7-row pass: a wider lookup pass took the old path,
# which pages card 0's dense weights in for every layer (~0.1-0.2 s a pass; 2026-10-24 A/B: see the README)
export STRATA_LOOKUP_TP="${STRATA_LOOKUP_TP-1}"
# keep-cache for every prompt batch (8192 -> 32768 on 2026-10-25, with patch-keepneed.py's xe sizing): the batch gives
# back only the VRAM it needs from card 0's expert cache, and the decode after it regrows the tail instead of
# re-seeding the whole cache (~0.9 s); a 16K batch keeps ~9 GiB of the 19.9 GiB cache (less streaming)
# (8192 from 01:40 to 02:30 on 2026-10-25: card-0 CAT errors after keep-cache shrinks - late promotions moved cached
# experts and the shrink freed their slabs; fixed in patch-pcrace.py / commit 1d9e8d3f, the reproduction ran clean)
export STRATA_SERVE_KEEP_MAX="${STRATA_SERVE_KEEP_MAX-32768}"
# the keep-cache shrink sized from what the last kept batch took on card 0 (its low point of free VRAM, sampled during it)
# instead of the calibrated 3.5 GiB + 0.45 MiB a column + 1 GiB: after a start's first batch the pools exist, and an
# 8K batch took 3.3 of the 8.0 GiB freed for it - the rest was card 0's experts evicted, streamed and re-uploaded for
# nothing.  Turns no deeper than the learned one (or 16K) only, 0.6 MiB a column more + 1.5 GiB margin; a low point
# under 1 GiB goes back to the calibrated size.  Regrow 8K 517 -> 266 experts, 16K 839 -> 582; fresh 8K / 16K -0.1 s
# (abwT67).  0 = the calibrated size always
export STRATA_SERVE_KEEP_LEARN="${STRATA_SERVE_KEEP_LEARN-1}"
# ... from the last 8 kept batches: the smallest VRAM a batch at least as wide took, x the depth ratio for a deeper
# turn (a chat's follow-ups end deeper every time, so the single learned size never applied to them), and the regrow
# after the batch left running while decode starts (published by the first TP pass after its copies): agent replay
# follow-ups 45.9 -> 45.0 s, fresh prompts and decode even (abwT102)
export STRATA_SERVE_KEEP_LEARN_HIST="${STRATA_SERVE_KEEP_LEARN_HIST-1}"
export STRATA_SERVE_REGROW_ASYNC="${STRATA_SERVE_REGROW_ASYNC-1}"
# the shared expert of a prompt call's parts of >= 4096 columns on the expert cards (card 0 bounds those batches):
# fresh 16K 7.26 -> 7.00 s, 31K 12.72 -> 12.09 s; KL within the run-to-run floor; smaller parts slower (abwT102)
export STRATA_SHEXP_TIER="${STRATA_SHEXP_TIER-1}"
export STRATA_SHEXP_TIER_MIN="${STRATA_SHEXP_TIER_MIN-4096}"
# a prompt's pageable arena experts reach the cards through small pinned rings that stay in the CPU's L3: each 1 MiB
# chunk is DMA'd as soon as it is copied, so the card reads it from the cache instead of DRAM (bench/dqb/stage_ring:
# 13.0 -> 21.7 GB/s onto cards 1-3); the prefetch's two 1.5 GiB pinned stages are gone, their budget is pinned arena.
# Agent replay prompts 127 -> 118.5 s, 16K / 31K from empty -2 / -1.4 s (2026-10-24 A/B: see the README)
export STRATA_XE_HOTRING="${STRATA_XE_HOTRING-1}"
export STRATA_XE_HOTRING_N="${STRATA_XE_HOTRING_N-2}"
# the prompt path's pageable arena experts written by the CPU straight into the expert cards' VRAM through their 32 GiB
# BARs (flex regions / stream buffers made through the xe uAPI, imported into Level Zero) instead of the hot rings'
# copy + a DMA a chunk: no ring DMAs queued behind the pinned chunks in the cards' one copy engine, the pinned DMAs and
# the CPU's writes side by side on the x16 uplink (~27 GB/s).  One stager (its default with this).  abwT83 against the
# base twice, fresh prompts: 512 2.00 -> 1.94 s, 2K 2.97 -> 2.90, 4.5K 4.14 -> 3.84, 8K 5.36 -> 4.94, 16K 7.83 ->
# 7.43, 31K 13.57 -> 13.19 (layers 20-39 1.4-2.2 -> 0.9-1.2 s); HOTRING_CHECK 40,000 experts, 0 differ (abwT80)
export STRATA_XE_BAR_STAGE="${STRATA_XE_BAR_STAGE-1}"
# the context is 1M tokens (larry-ds41.json; KV ~3 KB a token: the TP cards map ~3.1 GiB each as it fills): a side
# request inside a long conversation may write up to 262,144 positions and still park the conversation (~0.8 GB of host
# memory; was 65,536)
export STRATA_SERVE_PARK_MAX="${STRATA_SERVE_PARK_MAX-262144}"
# the TP cards' KV room beyond what 262K needs (0.8 GiB a card) holds experts until a long context needs it: then
# their slabs go back (copies read from the model file) - without it a 1M context cost 757 experts on the cards
export STRATA_XE_TP_EVICT="${STRATA_XE_TP_EVICT-1}"
# prompt calls (2026-10-25, abwT12-T20; README "Measured (2026-10-25 02:45 - 05:15: the dense card out of the prompt
# calls)"): in a pipelined prompt call card 0's part of the tier ran beside its own dense half and was the last card of
# ~90 % of the lower layers' calls (STRATA_XE_CALL_TL timelines).  DENSE_LITE: X read in place, no host waits, and no
# streamed experts on card 0 (in calls of >= STRATA_XE_DENSE_LITE_MIN tokens, default 0 = all); DENSE_INLINE:
# its part and the combine in the dense half's stream; DENSE_OFFLOAD: in calls of >= 4096 tokens card 0's cached experts
# are computed by the expert cards from their pinned arena copies; OVERLAP: the expert cards stream their overflow
# while they compute; GATHER_INIT: no memset of a card's part; SEED_TAIL_LAYERS: card 0's cache laid out with layers
# 20-39 first, so a batch's shrink gives back lower-layer experts (the upper layers are bound by the prefetch's staging)
export STRATA_XE_DENSE_LITE="${STRATA_XE_DENSE_LITE-1}"
export STRATA_XE_DENSE_INLINE="${STRATA_XE_DENSE_INLINE-1}"
export STRATA_XE_DENSE_OFFLOAD="${STRATA_XE_DENSE_OFFLOAD-4096}"
export STRATA_XE_OVERLAP="${STRATA_XE_OVERLAP-1}"
export STRATA_XE_GATHER_INIT="${STRATA_XE_GATHER_INIT-1}"
export STRATA_XE_SEED_TAIL_LAYERS="${STRATA_XE_SEED_TAIL_LAYERS-20}"
# the dense span's glue kernels (2026-10-25, abwT24-T27; README "Measured (2026-10-25 05:30 - 07:00: ...)"): hc_post
# reads the hc state once (it read it once per output), q's RoPE and the attention output's de-RoPE in place over the
# 64 rotated dims (they copied all 512), the DSV4 MQA prompt attention's Q conversion vectorised and its softmax a
# sub-group a row (three passes; the work-group form's two group reductions a row were the cost)
export GGML_SYCL_HC_POST_V2="${GGML_SYCL_HC_POST_V2-1}"
export STRATA_ROPE_INPLACE="${STRATA_ROPE_INPLACE-1}"
export GGML_SYCL_DSV4_MQA_QV="${GGML_SYCL_DSV4_MQA_QV-1}"
export GGML_SYCL_DSV4_MQA_SM="${GGML_SYCL_DSV4_MQA_SM-1}"
# a prompt batch's embeddings from the token table pinned at load (1.3 GB), staged f16 and uploaded async - the host
# converted them one thread from the mmap (0.4-0.6 s a fresh batch; 2026-10-25 abwT31)
export STRATA_EMB_PIN="${STRATA_EMB_PIN-1}"
# the pinned arena re-targeted at what prompts stream: card 0's cache below layer 20 (phase A streams it, the regrow
# refills it) + the next-ranked experts of layers 20.. (phase B's); the cache's layers 20.. (never streamed) pageable.
# A tier call's small uploads (row lists, slot maps, GEMM job / tile tables) by kernels reading pinned per-call copies:
# a memcpy waited in the card's one copy engine behind the staging DMAs (2026-10-25 abwT39-T50)
export STRATA_XE_PIN_STREAM="${STRATA_XE_PIN_STREAM-1}"
export STRATA_XE_ZCOPY="${STRATA_XE_ZCOPY-1}"
# the prompt attention of layers 0-19 (128 or 640 keys a query) as one oneDNN graph SDPA a 512-query span: keys 576 wide
# with an extra sink key (2026-10-25 abwT54; partial spans and layers 20-39 keep the oneMKL path)
export STRATA_ATTN_SDPA="${STRATA_ATTN_SDPA-1}"
exec "$XE_BIN/ds41_generate_xe" "${ARGS[@]}"
