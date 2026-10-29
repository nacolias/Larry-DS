> **Larry's build log, as kept on Larry** while the engine was built (newest sections first, 2026-10-19 to 10-27). Paths
> are Larry's: `/opt/ai/dsv41` (the checkout at `/opt/ai/dsv41/xolotl`), `/srv/dsv41` on the second NVMe drive.
> `larry-mode`, vllm-a, the studio, NPM and the public endpoint are Larry's own services. Here, `bench/...` and
> `patches/...` are this directory's `bench/` and `patches/`. `logs/...` stayed on Larry, except `logs/provenance/`
> (here `bench/provenance/`) and `qv/` (here `bench/qv/`). Commit hashes are this branch's. To set it up elsewhere,
> start at the [top-level README](../README.md).

# DeepSeek-V4.1-Flash-1017 on larry: `larry-mode ds41`

Created 2026-10-19. A fifth all-cards layout beside `big`, `strata`, `xestrata` and `small`:
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (763B-parameter MoE:
40 layers x 384 routed experts, top-6, Engram n-gram memory, 1M context) as the **GSQ mixed-tier quant**
[kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata)
(163 GiB: experts Q3_K / Q2_0 / ternary, dense Q8_0; no fine-tune, no abliteration - verified 2026-10-26 against
the official weights, see "Model provenance" below), served by
[Xolotl](https://github.com/Indras-Mirror/Xolotl)'s `ds41` engine (a Strata fork built for this model on one NVIDIA card)
with **an Intel expert tier written for larry**: four B70s, a RAM arena and the CPU pool. Same public endpoint, same API key.

```bash
larry-mode ds41            # stop the studio and vllm-a (or another all-cards server), start the ds41 server: ~80-95 s
larry-mode small           # back to vllm-a + the studio
larry-mode restart         # restart it with changed settings (larry-ds41.json, ds41-xe-engine.sh)
larry-mode status          # "mode: ds41 (...)" while it runs
```

Clients: Larry's public endpoint, a reverse proxy to the server (OpenAI: `/v1/chat/completions`; Anthropic: `/v1/messages`), model id
`deepseek-v4.1-flash-1017` (the server ignores the `model` field, so `larry` works too), the vllm-a API key. Thinking is on
by default and comes back as `reasoning_content`; `"reasoning_effort": "none"` turns it off (`low`/`medium`/`high`/`max`
or 1-100 are DeepSeek's numeric effort). Tool calls: DeepSeek's DSML, parsed into OpenAI/Anthropic tool calls.
Pictures (since 2026-10-26): OpenAI `image_url` parts and Anthropic image blocks (end to end so far: one a request).
Two requests at once (since 2026-10-27 19:04, `"parallel": 2`): since 20:47 their decode shares the passes - two long
answers at ~48 tok/s each, 92 in total since 23:08 (one alone ~71-85); a third waits for a free slot.
Sampling defaults: temperature 1.0, top-p 0.95. Context: 1,048,576 tokens (since 2026-10-24 20:35; see below).
Thinking that starts repeating itself is closed early and answered (since 2026-10-29 17:52: the server's loop detector,
`"reasoning_loop_recovery": "wrap"` in `larry-ds41.json`), not run into the thinking budget.

## Model provenance (verified 2026-10-26)

The GGUF's own metadata calls it "DeepSeek V4.1 Flash UNCENSORED (Strata v41 GSQ-RCO mixed tier)" with
`general.source.url` = dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8, while its model card says it is a quantization of
deepseek-ai/DeepSeek-V4.1-Flash-1017 with no fine-tune and no abliteration. The weights decide it
(`logs/provenance/prov2.py`, `prov3.py`, `prov3.log`; tensors range-read from both Hugging Face repos):

| check | vs official deepseek-ai/DeepSeek-V4.1-Flash-1017 (1bae0e10) | vs dealignai UNCENSORED-FP8 (d61c59ea) |
| --- | --- | --- |
| router weights, all 40 layers (F32 here, BF16 there) | bit-identical, every layer | layer 10: 3.6 % of values equal |
| token embedding, 96 sampled rows | identical (after BF16 -> F16) | 11 % equal |
| attention output projection wo_b, all 40 layers (Q8_0 here, FP8 there) | 0.54-0.55 % difference (Q8_0 noise) | layer 10: 11.3 % |
| shared-expert down projection w2, all 40 layers | 0.55-0.64 % | - |
| top singular value of the difference (an abliteration's rank-1 edit) | 0.016-0.13, flat spectrum | layer 10: 12.0 against 0.60 |

So the weights are DeepSeek's official 1017 release; the metadata strings are mislabeled. dealignai's repo is an
abliterated copy of the earlier DeepSeek-V4.1-Flash checkpoint (Sep 11; all 48 shards differ from 1017). Not compared:
the routed experts (quantized too coarsely for a clean test). Refusal behavior was not probed directly.

## Measured (2026-10-29 05:39 - 18:00: the morning's thinking loops - the build cleared at depth, a loop guard; where decode at depth goes)

**In the service since 17:52** (`bench/loop/launch-lw2.sh`'s window; `patches/patch-loopwrap.py` + `patches/svc-loopwrap.py`,
backup `larry-ds41.json.pre-loopwrap-20261029`): `"reasoning_loop_recovery": "wrap"` - thinking that repeats whole
passages is closed the way the thinking budget closes it, and the model answers.

* **What happened** (the engine log's `serve:` lines; no text was read): an agent session 04:12-06:09, 165 requests, the
  engine busy 51 min. Between 05:39 and 05:59 nine replies looped: each wrote ~24,600 tokens of near-verbatim repetition
  (21,000 of 21,050 lookup drafts accepted, 6.95 tokens a pass, ~205 tok/s) for ~2 min, until the 24,576-token thinking
  budget (max_tokens 32,768) closed the thinking; the next pass then read those 24,600 tokens again (8-9 s). The loop's
  text stayed in the conversation and the next turns looped too: 56K -> 155K in four loops, and again after the context
  had shrunk to ~100K and ~75K. 19.5 of the 51 minutes (38 %); the journal has no earlier budget wrap-up since Oct 19.
  The rest: replies 25.3 min (49 %; 84.7 tok/s at 40-155K), follow-up prompts of <= 3K new tokens 2.9 min (6 %), larger
  prompts 3.5 min.
* **Not the build** (`bench/loop/ltkl.sh`: the service's engine over its line protocol, fresh 96K and 150K prompts of
  code, the last 2,048 rows' logits dumped with `STRATA_SERVE_DUMP_TAIL`, 192 tokens greedy with `STRATA_TP_SYNC_CHECK=1`;
  `tailkl.py`): the engine as it runs, the same with the 10-28 changes off (`STRATA_ATTN_FSA`, `STRATA_XE_FN16`,
  `STRATA_XE_DQ2`, `STRATA_LI_TOPK`, `GGML_SYCL_TOPK2`, `STRATA_TP_AHEAD_S`), and as it runs again:

| fresh prompt | PPL now / again / changes off | KL now vs again (the floor) | KL now vs changes off | top-1 agreement |
| --- | --- | --- | --- | --- |
| 96K | 1.2318 / 1.2334 / 1.2317 | 2.51e-3 | 2.60e-3 | 99.07 / 99.51 % |
| 150K | 1.8950 / 1.8919 / 1.8996 | 6.41e-3 | 6.78e-3 | 98.19 / 98.14 % |

  Every TP replica check (216 regions after each prompt) found 0 rows that differ. The speculative accept is exact
  (`spec_accept`: a draft is kept with its probability, else the token is resampled without it), so the lookup drafts do
  not favor loops either: the loops are the model's own, fed by their text staying in the conversation.
* **The guard.** Strata's server had a loop detector (#728 / #1753; every 512 tokens: the share of the last 2,000 words
  inside 12-word passages seen three times >= 0.25, or a 2,048-character tail with a period of <= 64), off by default,
  and its two modes do not suit an agent at depth: "recover" reads the whole prompt again with the low-effort system
  sentence (~30 s at 100K, and the next turn reads the original again), "stop" ends the reply with no answer. "wrap"
  closes the thinking there with the budget's wrap-up, the same prompt and sampler (the engine keeps its prefix): a loop
  costs a few thousand tokens instead of ~24,600, and far less of it stays in the conversation. Tests:
  `serve/test_reasoning_loop_recovery.py` 22 (4 new: a passage loop, a one-word loop, novel reasoning left alone, batch
  slots), `serve/test_server.py`. Through the service (`bench/loop/loopprobe.py`: a request whose thinking is asked to
  write one line 300 times before it answers 17 x 23): the line 51 times, the guard at 4,608 tokens (coverage 0.296),
  then the wrap-up and "391".
* tsvc2 on the service (`logs/tsvc2-lw2.log`): every needle found (8K 1529 / 1912, 16K 2616, 31K 3380 tok/s), decode
  101.6 / 101.3 tok/s, essay 81.3 / 79.4, the tool call, the picture and its follow-up right, hidden RAM 34 GiB at most;
  concurtest PASS.
* **Where decode at depth goes** - the next item. The session's 97 replies of >= 200 tokens: 84.7 tok/s, 2.60 tokens and
  30.7 ms a pass, **15.4 CPU-pool experts a pass** (the essay bench: 5.8). A least-squares fit over them: ms a pass =
  17.4 + 0.50 x CPU experts a pass + 2.1 x tokens a pass - a CPU-pool expert is ~0.5 ms on the critical path, 7.7 of the
  30.7 ms. `bench/loop/agentd.py` (a deep agent session replayed: thinking on, temperature 1, a 64K first prompt, 24
  turns of the session's sizes) on a private server with `STRATA_XE_CPU_LOG` (`deeprun.sh dd1`): 107 tok/s, 3.32
  tokens and 31.0 ms a pass, 11.9 CPU-pool experts a pass (the short turns read as 7-row passes: 40 a pass, 45 ms);
  2,625 distinct experts outside the fixed placement, 4.9 uses each, 58 % of the CPU uses last used 17+ passes before;
  the promotion cache (380 slots) promoted and evicted ~6.5 a pass. Replayed (`bench/loop/pcsim2.py`) the cache's size
  is the lever: 380 slots 10.9 CPU experts a pass (the run 11.9), 600 6.9, 800 5.8, 1,000 5.2; a perfect-hindsight
  policy at 380 5.7. (A bigger cache that takes its room from the static experts did not pay on Oct 24.) The session's
  cards also gave ~665 static experts to the TP core's KV rows (seven 1 GiB slabs as the two slots grew to 262K-row
  buckets, 0.3-1.0 s each), and they do not come back when the conversations get shorter. A prompt that takes the
  prefetch (>= 2,048-3,072 new tokens) drops the whole promotion cache; smaller turns keep it.

## Measured (2026-10-28 18:20 - 19:10: big prompts, part 2 - the expert GEMMs, v2)

**In the service since 18:57** (`bench/longctx/xinstall.sh dq2` in a window, the engine before it in `build-xe/pre-dq2`;
`patches/svc-dq2.py`, backup `ds41-xe-engine.sh.pre-dq2-20261028`): `STRATA_XE_DQ2=1` (`patches/patch-dqv2.py` +
`dqv2_kernel.inc`). The same bits; with part 1, fresh prompts against the morning's service: **16K 6.48 -> 5.84 s, 31K
~11.0 -> ~9.9, 64K 21.3 -> 19.0, a 131K needle 45.4 -> 38.5 s** (-10..-15 %).

* **Where an expert card's call went** (abwT161: the service's engine under unitrace, `bench/speed/utdrive.py` - one fresh
  16K prompt traced after a 16K warm-up - and `utcall.py`, per call segmented at the X widening): an 8K-token call ran
  ~81 xe_dqgemm launches, ~35 ms of the card's GPU time (~29 TFLOPS), the X widening (FN16's peer reads over card 0's
  link) ~9 ms, the push ~4 ms, the row gathers and the deterministic sum ~7 ms, the shared expert (oneMKL) ~3.5 ms, the
  small experts' jobs ~3 ms. Bigger GEMM batches (`STRATA_XE_GEMM_RB` 2048 / 4096, abwT162): within the noise.
* **xe_dqgemm v2** (`bench/dqb/xe_dqgemm_v2.cpp` + `bench_dqgemm_v2.cpp`, `dqv2.sh`): v1's sub-groups each owned 16
  weight rows x all 128 tokens of a tile - the decoded weights in registers, every one of the 16 activation fragments a
  k-step read from local memory by hand (256 B a multiply). v2 keeps the tiles (128 rows x 128 tokens: the tier's tables
  as they are) but decodes each 64-value chunk of the 128 rows once into local memory VNNI-packed - B fragments by
  `joint_matrix_load` ext_intel_packed, the fast path - and gives a sub-group 32 tokens x 64 rows (4 x 4 accumulators)
  with its activation rows by 2D block loads from global memory: 128 B of local memory a multiply. The same decode, the
  same k order: **0 values differ** at every size. 96 experts at 128 tokens, planar: Q3_K **44 -> 63** TFLOPS, Q2_0 **51
  -> 89**, PTQ1_0 **48 -> 87**; the file layout (prefetched experts) 37 / 43 / 47 -> 50 / 62 / 77; 64 tokens 34 / 42 / 33
  -> 40 / 56 / 53; 160 tokens 37 / 45 / 39 -> 48 / 63 / 64; 32 tokens about even (abwT163).
* **In the engine** (abwT164, the same binary `STRATA_XE_DQ2` off / on): 16K 6.14 -> **5.84 s**, 31K 9.87 / 11.19 ->
  **9.67 / 10.12**, 64K 19.80 / 19.80 -> **18.82 / 19.15**, a 131K needle 40.75 -> **38.45 s** (found; its 12K turn 6.09 ->
  5.94 s). The tier waits fell (31K 3.82 -> 3.01 s, 131K 11.08 -> 8.43): card 0 bounds again.
* tsvc2 on the service (`logs/tsvc2-dq2.log`): every needle found (8K 1526 / 1918, 16K 2605, **31K 3433** tok/s), decode
  107.8 / 102.6 tok/s, essay 80.9 / 77.7, the tool call, the picture and its follow-up right, hidden RAM 34 GiB at most;
  concurtest PASS.
* **The front page's tables on this config** (abwT165 / T166, 19:20-19:30: `tabx.sh` fresh prompts from empty, two of
  each size; a 256K needle): 530 / 1,021 / 2,003 / 3,003 / 4,519 / 8,054 tokens in 1.90 / 2.17 / 2.77 / 3.20 / 3.49 /
  4.46 s (the faster of two), 16K 5.97-7.23 s, 31K 9.82-11.37 s and once 14.51 (that run's expert-card waits 6.0 s against
  2.5-3.9: big single prompts vary 10-15 % window to window - the same-window A/Bs above are the comparisons); the 256K
  needle 67.6 s (3,756 tok/s; Oct 26: 74 s), found, decode at that depth 87.4 tok/s, its 12K turn 6.2 s.

## Measured (2026-10-28 15:00 - 18:20: big prompts, part 1 - the prompt attention as one kernel that reads each query's keys where they lie; the expert cards' X in fp16 again)

**In the service since 17:59** (`bench/longctx/xinstall.sh fsa` in a window, the engine before it in `build-xe/pre-fsa`;
`patches/svc-fsa.py`, backup `ds41-xe-engine.sh.pre-fsa-20261028`): `STRATA_ATTN_FSA=1` (`patches/patch-fsa.py` +
`fsa_kernel.inc`), `STRATA_XE_FN16=2` again, `STRATA_LI_REMOTE` 131072 -> 0. Fresh prompts through a server (second
runs; the 984K needle after a 131K one, `bench/longctx/lt1m.sh`):

| fresh prompt | before (abwT157 fsa0) | + the fused attention (fsa1) | + the expert cards' X in fp16 (abwT158 fn2) |
| --- | ---: | ---: | ---: |
| 16K | 6.48 s | 6.28 s | **6.15 s** |
| 31K | 10.72 / 11.23 s | 10.15 / 10.77 s | **9.96 / 10.35 s** |
| 64K | 21.35 / 21.25 s | 20.09 / 20.27 s | **19.61 / 19.74 s** |
| 131K needle (abwT159 r0: 39.3 s) | 45.44 s | 43.96 s | - |
| 984K needle (the indexer on card 0 alone, abwT160) | 308.3 s (10-28 05:06) | 339.4 s (sharing as before) | **300.9 s** |

* **Where a big prompt's time went** (abwT146: `STRATA_GPU_PTL` + `STRATA_XE_CALL_TL` over fresh 16K / 31K / 64K
  prompts and a 131K needle, then `opprof-shapes.sh` on p30k): in layers 0-19 card 0's 512-column span took 5.6-5.9 ms
  (~180 ms a layer of a 16K batch) and the expert cards' two calls ~174 ms - still balanced. Of card 0's span the
  attention took 2.44 ms: DSV41_KGATHER 0.78 ms (each query's 656 key rows written, 387 MB), oneDNN's SDPA 1.10, its
  Q' / mask kernels 0.23, the op's output copy 0.33. An expert card's 8K-token call: X read 17.7 ms (168 MB f32 over
  card 0's link), GEMMs + gather 43-69 ms (~1.15 TFLOP: ~23 TFLOPS, xe_dqgemm alone ~47), the push 8-11 ms (fp16
  already, `STRATA_XE_COMBINE=2`).
* **No shared key set per block of queries** (abwT147: `patches/patch-dumpsel.py`, `STRATA_DUMP_SEL=1`, dumps every
  batch's selected rows; `bench/fa/selunion.py`): consecutive queries share about half of their 512 compressed rows -
  the union of Bq consecutive queries' selections is 1.2-1.5x of 512 at Bq 2, 1.5-2.2x at 4, 2.0-3.2x at 8, 2.6-4.7x at
  16 (code, 16-31K; prose - this log, 94K tokens - more: 3.9-4.7x at 16). An attention over each block's union (oneDNN's
  SDPA with q length Bq, `bench/fa/test_sdpa_blk.cpp`) would multiply the work; a kernel has to read each query's own
  rows.
* **The B70's operand paths for such a kernel** (`bench/fa/mqa_mb.cpp`, abwT148 / T149: a sub-group's 2 x 2
  accumulators, fp16): from local memory B col_major / row_major `joint_matrix_load` 13 / 27 TFLOPS (IGC gathers), a
  VNNI-packed copy 85, B by hand (a lane's 32 B of its row + get_wi_data) 78, A row_major 66; from global memory (2D
  block loads) B col_major 62, B row_major with the VNNI transform 73, A 73, both 51. Local reads ~3.7 TB/s; L2 24 MB.
* **The fused attention** (`patches/fsa_kernel.inc`; `bench/fa/test_fsa.cpp` against the service path - KGATHER + Q' +
  mask + oneDNN SDPA + the output copy - on the same inputs, and a double reference): a small kernel first writes the
  span's window rows in fp16 by position, so query t's 128 window keys are rows t .. t + 127 of it, read in place; then
  64 persistent work-groups (16 sub-groups, GRF 256), a query at a time: its Q to fp16 into the group's 192 KB slot of a
  scratch (reused query after query: it stays in L2), its keys in chunks of 128 - the window in place, its 512
  compressed rows converted into the slot - each chunk QK^T (2 x 2 accumulators a sub-group over the 512 dims; Q and
  K^T by 2D block loads), the chunk's max a head through local memory, P = exp(s - m) in fp16, PV into the query's O
  held in registers (4 x 4 accumulators a sub-group; V by VNNI-transforming block loads), and O / (l + exp(sink - m)).
  Five versions: v1 6.4 ms - IGC compiles a per-lane `vec<half, 8>` store as eight 2-byte scattered stores (the shader
  dump, `IGC_ShaderDumpEnable=1`), 256 such messages a chunk; v2 2.93 (sub-group block stores; both loops double-
  buffered by hand - IGC does not software-pipeline a load -> dpas loop); v3 2.58 (the window in place); v4 2.38 (block
  reads, the next chunk's rows prefetched to L1 - but 177 registers spilled); **v5 1.44** (Q's conversion by block IO too
  - its half stores were ~4000 messages a query - two rows in flight, P single-buffered: no spills). Against the service
  path: 30K **2.37 -> 1.44 ms** a 512-column span, early positions 2.03 -> 1.14-1.19, 130K with random rows (no L2
  reuse) 2.64 -> 1.84; rms error against the reference 5.07e-5 (the service path 5.19e-5). (v6, the softmax's reductions
  by transpose-reduce: 3.7 ms - not kept.)
* **In the engine** (`ggml_dsv41_attn_fsa`: a GGML_OP_DSV41_ATTN_SDPA node of form 2; with `STRATA_SDPA_O16`): card 0's
  dense phase of a 31K prompt 7.35 -> 5.79 s, 64K 14.6 -> 12.5 s, but the prompts only -3..-6 %: card 0 now waits on the
  expert cards (+0.5-0.9 s). The graph also drops the 387 MB gathered tensor. Numerics (`tlppl-ab.sh`, abwT157): p8k KL
  mean 2.92e-3 against a run-to-run floor of 2.35e-3 (p99 4.26e-2 / 3.49e-2, max 0.33 / 0.50, top-1 98.77 / 98.84 %) -
  oneDNN's SDPA in its day was 2.8e-3 against 1.9e-3; p30k PPL 1.7822 / 1.7777.
* **The expert cards' X in fp16 again** (`STRATA_XE_FN16=2`: card 0 converts the call's X once, the expert cards read
  it by peer loads; abwT158, all with the fused attention): 16K 6.37 -> 6.15 s, 31K 10.38 / 10.76 -> 9.96 / 10.35, 64K
  20.12 / 20.27 -> 19.61 / 19.74 (`=1`, the copy engine: between). It lost before (10-25, 10-26) while card 0 bounded.
  Numerics of both (abwT160): p8k KL 2.76e-3 against 2.62e-3 (top-1 98.86 / 98.91 %), p30k PPL 1.7809 / 1.7811.
* **The long prompts' indexer back on card 0** (`STRATA_LI_REMOTE=0`): with the fused attention the 984K needle got
  slower while sharing - 339.4 s at card 0's 35 % (card 0's dense phase 235.2 -> 206.7 s, its waits on the expert cards
  58.7 -> 110.9 s: the share's kernels slow the cards' expert calls), 357.0 s at 50 %, 306.5 / 308.4 s at 65 %, 307.7 s
  without it, 300.9 s without it and with FN16=2. 131K: 39.3-40.7 s (a fresh engine's first big prompt once 46.7).
* **Open: one engine went silent after its 984K batch** (abwT159 s65: the fused attention, the indexer shared with card
  0 keeping 65 %): the batch finished (306.5 s), the engine log ends after "prompt phases" (DSpark tail 0), the client's
  request failed with the server closing the connection; no GPU fault, no OOM; that server's log was overwritten
  (`lt1m.sh` keeps them now). The same config again (abwT160 s65b) passed (308.4 s, its turn 8.7 s). The service runs
  without the share.
* **Taller expert GEMM tiles** (`bench/dqb/dqmt.sh`: xe_dqgemm with 160 / 192 token rows a tile, abwT158): +10-25 % for
  experts of 144-160 tokens (PTQ1_0 38.4 -> 49.0 TFLOPS at 160), slower at <= 128 and >= 192 - a per-expert choice would
  buy ~2 ms a call; `patches/patch-dqmt.py` written, not applied.
* tsvc2 on the service (`logs/tsvc2-fsa.log`): every needle found (8K 1390 / 1835, 16K 2541, **31K 3323** tok/s - part
  2's 3108), decode 84.3 / 104.9 tok/s, essay 79.6 / 76.2, the tool call, the picture and its follow-up right, hidden
  RAM 34 GiB at most; concurtest PASS.

Next for big prompts (after part 2): card 0 bounds again - its span's remaining ~4.3 ms (the fused attention 1.44, the
weight GEMMs ~1.5, the hc steps ~1); at 1M card 0 bounds (the indexer).

## Measured (2026-10-28 02:50 - 05:10: long contexts, part 2 - the 1M prompt's indexer: its top-k in two stages, its queries shared with the expert cards)

**In the service since 05:06** (`bench/longctx/xinstall.sh lt` in a window, the engine before it in `build-xe/pre-lt`;
`patches/svc-topk2.py`, `svc-lif1.py`, `svc-litopk.py`): a fresh 967K prompt **338.9 -> 308.3 s** - with part 1 the
time to its first token **352.6 -> 308.9 s** (-12 %). Every answer the same as before: the top-k picks the same set, the
indexer's ids are a top-k of card 0's own scores on every checked call.

| a fresh 967K prompt after a 131K one (`bench/longctx/lt1m.sh`) | prompt | card 0's dense phase | waits on the expert cards |
| --- | ---: | ---: | ---: |
| before part 2 (the radix top-k; `r1m-ra.sh`, 10-28 01:50) | 342.7 s | 294.9 s | 31.9 s |
| the top-k in two stages (the service since 03:12) | 338.9 s | 282.4 s | 38.9 s |
| + the indexer's queries shared with cards 1-3, their copies bf16 | 313.2 s | 231.9 s | 65.7 s |
| - the same, their queues at low priority | 322.3 s | 248.4 s | 58.4 s |
| - their copies fp16, card 0 keeping 25 % | 309.5 s | 230.0 s | 62.6 s |
| - their copies fp16, card 0 keeping 35 % (**the service**) | **308.3 s** | 235.2 s | 58.7 s |

* **The top-k in two stages** (`patches/patch-topk2.py` + `topk2_kernel.inc`, `GGML_SYCL_TOPK2=1`): ggml-sycl's radix
  select reads each query's whole score row up to five times (at 512K rows, 4.5 ms a call of 256 queries: the card's
  read bandwidth). Now a work-group reads its row once for its blocks' largest keys (8192 blocks at most, in SLM), takes
  the k-th largest of those - T, its exact key - and runs the radix select and its emit over the blocks whose maximum
  reaches T only: k blocks hold a key >= T each, so the row's k-th largest key is >= T and every key at or above it lies
  in such a block - the same set (ties at the pivot are taken in no fixed order, as before). `bench/li/topk2_test.cpp`
  (`topk2.sh`): 1280 rows of four kinds (normals; relu-weighted sums with many exact zeros; -inf past each query's
  visible rows, fewer visible than k too; few-level ties) at 64K-1M columns all the same; 256 rows of 512K columns
  4.32 -> 1.67 ms (2.6x), of 1M 8.83 -> 2.94 ms (3.0x). A 512K needle's dense phase 136.3 -> 130.5 s.
* **The fused indexer reading its BF16 keys in place** (`patches/patch-lifbf16.py`, `GGML_SYCL_LI_FUSED=2`): mode 1
  converts the whole key table to fp16 on every call; mode 2 runs the same kernel in bf16 on the keys as stored. The
  same bits (the values are FP4-rounded: exact in both; `GGML_SYCL_LI_FUSED_CHECK16` and `bench/li/lifbf_probe.cpp`: 0
  scores differ) - but bf16's kernel is slower at depth: 2-4 % faster up to 64K keys, even at 128K, 3 % / 6 % slower at
  256K / 512K (19.0 against 20.3 ms for 512 queries). In the service 03:12-04:05, then back to mode 1
  (`svc-lif1.py`). (tsvc2's 16K needle at 1912 tok/s that hour was noise: at 8K keys mode 2 is the faster one.)
* **The indexer and its top-k as one op, its queries shared with the expert cards** (`patches/patch-litopk.py`,
  `patch-litopk-eng.py`; `STRATA_LI_TOPK=1`, `STRATA_LI_REMOTE=131072`, `GGML_SYCL_LI_REMOTE_SHARE=35`):
  `ggml_lightning_indexer_topk` (LIGHTNING_INDEXER, op_params[0] = 2) gives I32 [k, n] straight from the scores - the
  [cap, n] score tensor and its column slices are gone from the graph - for the layers that rank every visible row (2 /
  8 / 14). For a served prompt of >= 131072 tokens the engine prepares cards 1-3 before its first batch (each a fp16 copy
  of the three layers' keys sized for the prompt's last cap bucket, score scratch for its share of a span: 657 MiB a
  card at 1M) and frees them after it. A call of >= 64K keys keeps 35 % of its queries on card 0; card 0 stages the
  keys' rows the others lack and their queries in a plain buffer of its own (card 0's KV pages and ggml's pool are never
  read by a peer), each card pulls its part, converts it to fp16, scores and ranks it on a queue of its own, and card 0
  pulls their ids back after its own share - in queue order, no host wait.
* **Its first build was wrong** (a test server only): every query on cards 1-3 got wrong ids from its second call on
  (`GGML_SYCL_LI_TOPK_CHECK`: each query's ids against card 0's scores of every query - distinct, in range, a top-k of the
  row). The cards' copy was counted as complete up to the call's cap bucket, but the bucket's later rows were not
  written yet: the next span wrote them and nobody copied them. Now the op carries the rows already final (its last
  query's (position + 1) / ratio) and the cards copy up to that; 14 checked calls since, 0 queries wrong.
* **Why only -30 s of card 0's -47**: cards 1-3 also run every layer's experts for the prompt, and the indexer's share
  now competes with those calls - the dense half waits for them 20 s longer. Their queues at low priority only moved the
  wait (card 0 then waited for the shares instead: 322.3 s); fp16 copies (6 % less work there) and a larger share for
  card 0 helped a little. The needles (131K, 967K) found on every run, no faults; decode at depth unchanged (68.9-70.5
  tok/s against 69.7).

## Measured (2026-10-28 01:20 - 02:50: long contexts, part 1 - the wait before the first reply after a long prompt; where a 1M prompt's time goes)

**In the service since 02:33** (`bench/longctx/xinstall.sh ra` in a window, the engine before it in `build-xe/pre-ra`;
`patches/patch-roomahead.py`): after a long prompt the first reply comes almost at once.

| the first reply after the prompt (the needle's 9-token decode) | before (10-27, `r1m-stage.sh`) | now (`bench/longctx/r1m-ra.sh`) |
| --- | ---: | ---: |
| a 131K needle (its conversation's KV replicas 1K -> 262K rows: a slab a card) | 2.22 s | **0.33-0.36 s** |
| a 967K needle (-> 1M rows: three slabs a card) | 8.09 s | **0.72 s** |
| - its cards' evictions | 1.15 + 4.97 + 0.99 s, in turn | 2-5 ms, side by side, read ahead |
| - its replicas' sync | 7.89 s | 0.44 s |

* **What the wait was**: the first pass after a long prompt grows the conversation's KV replicas on the expert cards to the
  prompt's cap bucket, and each card first gives VRAM back - its evictable last slabs go back to the arena with the
  experts' host copies read from the model file then (`STRATA_XE_TP_EVICT`): one card after another, ~0.6 s a GiB and
  worse under host memory pressure (card 2's 5 s). Then the rows were copied twice (the growth copied them from card 0,
  the sync again: 0.34 s at 967K).
* **The read-ahead** (`STRATA_TP_AHEAD_S`, 30): once a prompt's batches left look shorter than 30 s, the engine tells the
  TP core where the prompt ends, the core tells the tier each card's coming growth, and the tier works out the slabs
  `evict_slabs` would free and reads their experts on a helper thread (2 readers; `STRATA_XE_TP_AHEAD_MEM_GIB`, 8, of
  MemAvailable must stay over), re-checked at every batch. The eviction takes a slab's copy when its experts are the ones
  read; what the growth does not take is dropped when the replicas reach the target (or at the next request).
* **Side by side** (`tier.tp_make_room_multi`): the cards' evictions a thread each (their bookkeeping under a lock), the
  driver waited for once a round of slabs instead of once a slab; the sync copies only the rows below where its own
  growth began.
* **What it costs the prompt: nothing measurable** (`bench/longctx/ra-ab.sh`: the same engine, two fresh 131K needles
  each - conversation slots 1 and 0): read-ahead off 48.44 / 40.98 s, on 49.41 / 40.97 s; the first reply 3.52 / 1.41 s
  against 0.35 / 0.36 s. Without it the side-by-side evictions alone took 3.1 and 1.1 s - the fresh host memory, not the
  reads. The first big prompt after an engine start is the slow one either way (its expert phase 14 s against ~8;
  `bench/speed/vmwatch.sh`: kswapd, direct reclaim and compaction all through it), which my first reading took for the
  read-ahead's cost. At 967K: 342.7 s (344.5 before) and once 363.0 s (its expert phase 47 s against ~32; not seen again).
* **A deadlock the first test found** (a test server, killed; no user request): the tier's `init()` calls `close()` first,
  and that set the helper's stop flag for good - its thread quit at once and the evictions waited for reads that never
  came. Fixed; an eviction now waits for a read at most `STRATA_XE_TP_AHEAD_WAIT_S` (60 s), then reads it itself.
* tsvc2 on the service (`logs/tsvc2-ra.log`): every needle found (8K 1333 / 1750, 16K 2453, 31K 3110 tok/s), svcbench
  100.9 / 96.3, essay 78.8 / 77.1 tok/s, the tool call and the picture right, hidden RAM 34 GiB at most;
  `concurtest.py` PASS. The 967K replica checks (`STRATA_TP_SYNC_CHECK=1`): 0 regions differ.

**Where a 967K prompt's time goes** (`bench/longctx/prof1m.sh`: card 0's ggml-sycl time op by op, `GGML_SYCL_OP_PROF`'s
cumulative dumps every 250K marks split by `bench/longctx/profdiff.py`; that run's prompt 346.9 s):

| card 0, s (the prompt's tokens) | 0-180K | 180-360K | 360-524K | 524-704K | 704-868K |
| --- | ---: | ---: | ---: | ---: | ---: |
| card time | 49.1 | 53.5 | 58.4 | 63.8 | 67.8 |
| `LIGHTNING_INDEXER` (the prompt indexer: layers 2 / 8 / 14 run every token) | 1.8 | 5.5 | 8.7 | 12.5 | 15.8 |
| `TOP_K` (each query's 512 best visible rows) | 1.0 | 3.3 | 4.6 | 8.6 | 9.0 |
| weight GEMMs / oneDNN attention / key gather / `HC_FA` | 9.4 / 7.5 / 5.1 / 4.7 | 9.4 / 7.6 / 5.3 / 4.6 | 9.4 / 7.7 / 5.6 / 4.7 | 8.9 / 7.2 / 5.3 / 4.4 | 8.9 / 7.2 / 5.4 / 4.4 |
| idle before a graph | 7.7 | 6.5 | 6.4 | 6.1 | 6.4 |

Only the indexer and its top-k grow with depth: ~86 s of the prompt's ~347 together, 37 % of the deepest interval. The
indexer runs near the XMX peak (~137 TFLOPS: 256 queries x 512K keys x 32 heads x 128 in ~8 ms a call - past 256K rows
the slices halve the queries, two calls a span) and converts the whole key table to fp16 on every call (~0.45 ms of the
8); the top-k reads each 2 MB row up to five times (~4.5 ms a call: the card's read bandwidth).

## Measured (2026-10-27 21:00 - 23:10: card 0's KV mapped on demand, in the service - the other cards' reads of its later pages; the CPU pool's ternary rows)

**In the service since 23:08** (`patches/svc-lazykv.py`, backup `ds41-xe-engine.sh.pre-lazykv-20261027`): card 0's KV
state mapped as the conversations grow (`STRATA_DENSE_LAZY_KV=1`, 65,536 positions of each conversation slot at load:
0.84 instead of 6.28 GiB) - the rest of card 0 holds experts. On the same engine (`abw-lazyspeed`):

| | both slots' KV at load | mapped on demand |
| --- | ---: | ---: |
| the agent replay (`agentx.py`) | 98.8 s (prompts 66.7, decode 30.7) | **90.5 s** (prompts 58.7, decode 30.4) |
| CPU-pool experts a decode pass (the replay) | 7.5 | 5.7 |
| fresh 2K / 4.4K / 8K / 16K | 3.43 / 4.05 / 4.93 / 6.54 s | 3.26 / 3.85 / 4.97 / 6.56 s |
| decode: essay / code / math | 76.3 / 119.2 / 115.0 tok/s | 84.6 / 118.5 / 121.8 tok/s |
| two long answers at once (the service, 800 tokens each) | 43.5 + 43.2 = 82.2 tok/s | 47.6 + 48.7 = **92.5** tok/s |

tsvc2 on the service (`logs/tsvc2-lazy.log`): every needle found (8K 1519 / 1762, 16K 2459, 31K 3134 tok/s), svcbench
103.3 / 101.3, essay 80.6 / 77.9 tok/s, the tool call, the picture and its follow-up right, hidden RAM 34 GiB;
`concurtest.py` every answer right.

* **Why it garbled a 984K conversation** (`patches/patch-tpsynclog.py`, diagnostic: `STRATA_TP_SYNC_LOG=1` logs every
  replica sync, `STRATA_TP_SYNC_CHECK=1` compares sample rows of card 0's KV with each replica right after it): the TP
  core's replicas on cards 1-3 came out different from card 0 exactly in the rows card 0 holds in a physical allocation
  mapped after the first one (KV_INIT's pages end at ratio-1 row 133,120, ratio-2 row 67,584) - 24 of 216 sampled
  regions after a 131K conversation's first turn already, 48 after the 984K prompt (its decode then 254 ms a pass, DSpark
  1.26 tokens a pass, and faults on three cards at the next request). The replica's own rows all lay in one allocation,
  so the other cards' READS of card 0's later pages were wrong. Alone the same mapping order reads right
  (`bench/conc/vmm_span_probe3.cpp` - one allocation of 64 MiB to 1 GiB, filled by a kernel or a copy - and
  `vmm_span_probe4.cpp` S1 - the engine's order: map, peers read, map more, write, peers read, in one copy across the
  boundary or split: all exact; `vmm_span_probe2.cpp`'s "failure" was its own pattern bug). What in the engine breaks it is
  not found - not BAR staging (`bench/conc/lazyab.sh`: the same 24 regions without it); the pager could not be switched
  off (card 0 then runs out of VRAM).
* **The fix** (`patches/patch-tpstage.py`): the other cards no longer read those pages. Card 0 copies the compressed-KV
  and indexer-key rows (and the spare rows) into a plain 64 MiB device buffer of its own and every other card pulls them
  from there, a chunk at a time - in the sync after a prompt and in the replicas' growth (`hooks.kv_stage`, set when card
  0's KV is mapped lazily). With it every sync check finds 0 regions different; a 131K needle in one slot and then a 984K
  one in the other are both found, the 984K's 400-token decode at 68.5 tok/s (32 ms a pass; allocated at load: 70.1),
  its 12K turn 12.45 s, no fault (`bench/conc/r1m-stage.sh`). The first reply after a ~1M prompt still waits ~8 s for the
  replicas' sync (6.7 s unstaged) - that is optimization #5's.
* **The CPU pool's ternary rows** (`patches/patch-ptq1rows.py`): `bench_cpu_pool` (one cold expert a layer on 6 workers +
  the host, against a streaming read of the same bytes on the same threads): Q3_K and Q2_0 already stream at the DRAM
  ceiling for one token (33-36 GB/s; 2-3 tokens 7-24 % below it), PTQ1_0 does not - 22.0 / 16.8 / 14.7 GB/s at 1 / 2 / 3
  tokens against 30.3. Its decode is now done in 8-bit lanes (x = byte ^ 0x80: multiplying by 3 mod 256 keeps the ^ 0x80,
  so trit n of all 26 code bytes is two compares of 3^n x mod 256) and each token's fp16 scales are converted once a call,
  not once a row: **27.1 / 20.3 / 17.5 GB/s (+20-23 %), bit-identical** (`bench/cpu/ptq1_rows_bench.cpp` compares every
  output with the engine's kernel; the engine's entry point old vs new wrote identical files). The per-token half keeps
  its exact order - it sets the engine's rounding. `STRATA_PTQ1_ROWS1=1`: the old decode.
* **The CPU pool in decode** (`bench/conc/cpulog.sh` + `cpulog_stats.py`, `STRATA_XE_CPU_LOG`): a pass alone puts 5.8
  experts on the pool (94 % of them for one token) - 74 MB of weights in 2.6 ms; two at once 18.9 (90 % one token) - 239
  MB in 9.4 ms: 25-29 GB/s, what the host's memory gives. Faster kernels buy decode little; more experts on the cards (the
  lazy map's VRAM) or fewer bytes a cold expert do.
* `bench/conc/s2build.sh` now puts back the files the service ran when the build started (it put back the PRE-stage-2
  engine after every build: the 21:46 restore started stage 1 for a minute, unnoticed until the checksums).
* **What an agent's follow-up prompt waits on now** (23:15-23:45, the agent replay; `STRATA_XE_CALL_TL` timelines through
  `bench/speed/calltl_stats.py`, `STRATA_XE_LOCK_PROF`): the tier's calls stream next to nothing to the cards any more
  (0-0.9 experts a card a call) - they are host-bound. A ~380-token call: 21.9 ms, its CPU-light part (23 experts) 19.7
  ms and last 60 % of the time; card 1 does 21 GEMM experts (621 rows) and 32 small-expert jobs - ~0.5 GB of VRAM reads,
  2-3 ms of GPU work - in 16.3 ms of its thread. ~1,500-token calls: 34.7 ms, the cards last (~21 ms), the CPU part 5.6
  ms. Per 400 calls the submit lock is waited for ~8.8 s (summed over the threads) and held ~2.5 s - ~5 ms of each card
  thread's call; one site (`hot_put`, a take per 1 MiB chunk of the prefetch's copies) takes it ~170 times a call.
  Knobs, the agent replay's prompts: base 60.0 s, `STRATA_XE_ZCOPY=2` 61.3, `STRATA_XE_CPU_LIGHT=4` 60.9, `=2` 60.1,
  `STRATA_XE_GEMM_MIN=4` 61.5 - all within the noise. What is left there is the calls' structure (far fewer enqueues
  and lock takes a card, no lock where nothing pageable is copied) - not a knob.

## Measured (2026-10-27 19:20 - 20:58: concurrent sessions, stage 2 - the two sessions' decode in shared passes)

**In the service since 20:47** (`bench/conc/s2install.sh` in a window; the engine before it kept in
`build-xe/pre-s2`). When both slots decode, each plans its DSpark round in its own conversation, ONE TP pass verifies
both rounds' rows, and each accepts against its own rows. Two long answers at once now decode at 43.5 + 43.2 = 82.2 tok/s
(stage 1, taking turns: 35.6 + 35.8 = 71.4); a request alone runs as before.

* **The measurement first** (`patches/patch-widthbench.py`, diagnostic: `STRATA_TP_WIDTH_BENCH=R` - after each request
  of a test server, R passes of every width 1..7 through `pass_v`; `bench/conc/wb.sh`, `wbessay.py`). Rows of two real
  conversations in one pass (the conversation's own last tokens + the previous request's) cost what one conversation's
  rows of that width cost; random token ids were a bad proxy (they route to cold experts). The solo round in the same
  run: 2.0-2.16 rows verified, the verify pass 19.1-20.7 ms, DSpark's step 2.2-2.4 ms. The model said ~+55 % over
  taking turns, over the +25 % bar set for building it - too optimistic (below).

| median ms a pass (8K context) | 1 row | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one conversation's own rows | 13.5 | 17.2 | 18.5 | 21.6 | 24.0 | 26.7 | 29.9 |
| two conversations' rows (half each) | - | 17.3 | 18.4 | 21.1 | 23.8 | 26.1 | 29.2 |
| random token ids | 16.0 | 26.1 | 27.8 | 30.5 | 31.1 | 38.1 | 41.8 |

  (At 31K the same: own 13.6 / 17.6 / 18.9 / 21.8 / 24.0 / 26.7 / 29.6, two conversations' - / 17.5 / 18.5 / 21.7 /
  24.6 / 27.0 / 29.5.)
* **One pass, several conversations** (`patches/patch-batch-tp.py`, `patch-batch-kern.py`): `Ds41Tp::pass_multi` takes
  runs - consecutive positions of one conversation each, rows back to back, <= 7 in all. The weights' work (every
  mat-vec, the shared and routed experts, the CPU pool, the LM head) takes all rows at once; the state kernels (the
  window row, the compressor, the indexer's scan and top-k, the attention) run once a run on views from the run's
  first row, each in its own conversation's state (only attn_dec's grouped q8 output needs the run's place in the
  pass: `row0` / `n_rows`). A plain pass is the one-run case of the same code (offsets 0: the same kernels with the
  same arguments). A pass of runs has its own recorded graphs (keyed by the runs' conversations, widths and scan
  buckets; recorded at first use). Each run's Engram rows come from its own conversation's n-gram history
  (`patch-batch-dense.py`).
* **The serve loop** (`patches/patch-batch-serve.py`): a DSpark round is split into plan (emit t, the drafts, the
  verify rows) and accept (against the pass's rows, DSpark's step on the run's taps, emit the accepted drafts). With
  two slots decoding, each plans (the drafts capped so every row fits the 7), one `pass_multi` runs them, each accepts;
  `STRATA_SERVE_JOINT=0` makes them take turns again. A request alone runs plan, its own pass, accept, as before.
* **Checked** (`patches/patch-batch-check.py`, diagnostic: `STRATA_SERVE_JOINT_CHECK=1` runs every run's rows through
  a pass of its own first - its ratio-2 compressor state kept aside and put back - and compares the joint pass's
  logits with them): KL(own || joint) 0.0006-0.008 a row, the same argmax in 96.5-100 % of the rows. The baseline
  (`=2`: a conversation's rows against the same rows with one more row after them, no joint pass) gives 0.0012-0.0015
  and 98.7-100 % - the joint pass differs from a conversation's own pass as much as two widths of its own pass do
  (the kernels' forms go by width; DSpark's verify widths vary the same way every day). With the expert promotions
  frozen (`STRATA_TP_PROMOTE_CHUNKS=0`) a request alone is byte-deterministic: two runs of a 400-token answer
  identical. `concurtest.py`: every answer right (two needles read and decoded side by side, both follow-ups from
  their own slots); the tool call right. Tests: `bench/conc/s2test.sh` (the staged engine: `s2build.sh` builds into
  `/opt/ai/dsv41/s2bin` and puts the service's files back), `jointcheck.py`.

| two long answers at once | taking turns | one pass |
| --- | ---: | ---: |
| test server, the same engine (`STRATA_SERVE_JOINT=0` / 1), 1,500 tokens each | 37.4 + 38.9 = 74.6 tok/s | 43.9 + 48.2 = 83.7 tok/s (+12 %) |
| the service, 500 / 800 tokens each (stage 1 / stage 2) | 35.6 + 35.8 = 71.4 | 43.5 + 43.2 = 82.2 (+15 %) |
| one answer alone | 72-81 | 71-81 |

* **Why not +55 %: the CPU pool.** With `STRATA_TP_PROF=1` a joint pass (3.9 rows) spends 9.5-10.5 ms in the CPU
  pool's experts on 13-14 of its layers, a pass alone (2.1 rows) ~3 ms on 5-6: two conversations' hot experts do not
  fit the cards' caches together, more picks land in the RAM arena, and each such layer waits for the host. The width
  bench missed it: it repeated identical passes, and the tier promoted their experts into VRAM before the median.
  `STRATA_XE_CPU_LOG` (21:02, `bench/conc/cpulog.sh`, 400-token answers): a pass alone puts 5.8 experts on the pool
  (94 % of them for one token) - 74 MB of weights in 2.6 ms; two at once 18.9 experts (90 % one token) - 239 MB in 9.4
  ms. The pool streams 25-29 GB/s, close to what the host's memory gives: faster kernels would buy little here; fewer
  bytes a cold expert (a lower-bit RAM tier) or more of both conversations' experts on the cards would. One format
  class (15.2 MB blobs, 95 % pageable) is half the picks.

tsvc2 on the service (`logs/tsvc2-s2.log`): every needle found (8K 1446 / 1770, 16K 2457, 31K 3106 tok/s), svcbench
84.0 (the first after the restart) / 96.5, again 97.9 / 92.8, essay 75.0 / 73.3 tok/s, the tool call, the picture and
its follow-up right, hidden RAM 34 GiB.

**Limits now.** Two sessions decode together at ~43 tok/s each (48 since the lazy map, next section). A prompt being read
still pauses the other session for its whole layer-major batch (up to 16K tokens, ~6 s). `BYIELD` is not acted on.
Pictures are read alone. A third request waits for a free slot. Both slots' card-0 KV state is allocated at load (mapped
on demand since 23:08: next section).

## Measured (2026-10-27 17:30 - 19:20: concurrent sessions, stage 1 - two conversations at once)

**Two sessions at once, in the service since 19:04** (19:12 with the slot pick below; `patches/svc-slots.py`:
larry-ds41.json `"parallel": 2`, backup `larry-ds41.json.pre-slots-20261027`; both slots' card-0 KV state allocated at
load, +3.1 GiB on card 0). The server
already had Strata's batch-slot client (`generate_batched`); the engine now speaks its lines. A request alone runs as
before (GEN, DSpark); when a second one arrives the server stops the first, re-admits it into a slot (its prompt plus
what it wrote: the engine continues from the state it holds) and admits the new one (`BGEN` -> `BADM`), and the two
take turns until each ends (`BT` / `BDONE`).

* **Conversation slots** (`patches/patch-seqs-tp.py`, `patch-seqs-dense.py`, `patch-seqs-gen.py`): the TP core, the dense
  half and the Engram history hold N conversations side by side - per conversation the window rings, compressed KV and
  indexer keys (card 0's in the lazy map when that is on, one set of units a conversation), compressor states, snapshot, written
  range, the KV replicas on cards 1-3, DSpark's draft rings and the TP core's recorded graphs (they bake the state's
  addresses: pre-recorded per conversation at attach, 28 graphs a card in ~1 s). `bind_seq(s)` swaps them in between
  passes - no copy, no re-recording. A park now holds (layer, kind) regions, so a parked conversation comes back into
  either slot. A request goes to the slot whose conversation it continues best, else an empty one, else the one used
  longest ago (its conversation goes to the pool first).
* **The protocol** (`patches/patch-slots-serve.py`): the decode loop became a state with one round a call (DSpark: emit,
  verify the drafts, accept; or one plain step); the request alone loops it as before, the slots' windows take one round
  each in turn. Every pass is still a single-sequence pass in its own conversation slot, so the kernels are unchanged and
  each conversation decodes as it would alone. Windows run whenever no control line waits, and between the layer-major
  batches of a prompt being read. `BYIELD` is accepted and not acted on (the read finishes).
* **Card 0's KV state mapped on demand, fixed** (`patches/patch-kvbucket.py`, `patch-kvresident.py`): the 2026-10-24 and
  17:34 DEVICE_LOSTs were ONE peer copy spanning two physical allocations of the VMM reservation - the TP core's replica
  growth after a 131K prompt read rows [8192, 262144), mapped partly at load and partly just before the prompt
  (`bench/conc/vmm_span_probe.cpp`: split at the boundary exact; one copy across with the new allocation made resident
  on the peer exact; one copy across without - card 1's copy engine faulted, -ENOENT). The driver makes only a copy's
  first allocation resident on the peer by itself. Every newly mapped range is now made resident on the other cards,
  and the map follows the readers' power-of-two cap buckets - so a second slot's 1M would cost VRAM only when used,
  but the lazy map is not in the service yet (see 1M below).
* **A live rewind** (`patches/patch-rewind.py`): a prompt that diverges from the live conversation's last prompt within
  its last 16 positions (a re-rendered tail) resumes there after `restore()` (the window ring at the snapshot holds the
  144 positions before it) - the live counterpart of the partial unpark. reusetest: 9 tokens read instead of 2,439.
* **The slot a request goes to** (`patches/patch-seqpick.py`): the one whose conversation it continues (extends it,
  reaches past its snapshot, or rewinds into it), at any length; else an empty one, else the one used longest ago.
  The first form wanted a 2,048-token match: a short chat's follow-up went to the other slot and read its prompt
  again (tsvc2's tool turn 0.7 -> 1.9 s at 19:04).
* The text log (`STRATA_TEXT_LOG`) numbers each request (`patches/patch-textlog.py`).

The tests (`bench/conc/`): `seqtest.py` (two conversations resident), `concurtest.py` (requests at once through the
server), `twoessays.py` (two long answers at once), `reusetest.py` (the pool, the rewind), `s1test.sh` (a private
server: all of them + needles + the tool call), `r1m.sh` (long needles in either slot, with `STRATA_ADDR_LOG`).

| two resident conversations (s1a, a private server, no protocol yet) | read | reused | time |
| --- | ---: | ---: | ---: |
| X (31K needle) and Y (6.6K needle), first turns | 31,157 / 6,640 | 0 | 10.5 / 3.9 s |
| their next turns, alternating X, Y, X, Y | 19 each | the whole conversation | 0.3-0.5 s |
| Z (a new 6K prompt): the slot used longest ago (X's) - X into the pool | 6,342 | 0 | 3.9 s |
| X again (from the pool, Y's conversation pooled for it), then Y again (from the pool) | 29-31 | 31,216 / 6,675 | 0.4-0.7 s |

`concurtest.py` on the service (19:14, `"parallel": 2`), every answer right:

| | request | first token | done | tokens |
| --- | --- | ---: | ---: | ---: |
| round 1 | a 600-token essay, alone | 0.4 s | 12.3 s | 600 |
| | a 4K-token needle arriving 3 s into it | 4.1 s | 4.3 s | 9 |
| round 2 | the essay's follow-up, an 8K and a 12K needle at once (two slots) | 0.2 / 4.8 / 10.2 s (the 12K waited for a slot) | 11.5 / 5.0 / 10.4 s | 118 / 8 / 9 |
| round 3 | both needles' follow-ups at once | 0.4 / 0.8 s (8,129 / 11,612 reused) | 0.9 s | 11 / 12 |

`twoessays.py` on the service: one essay alone 72.4 tok/s; two at once 35.6 + 35.8 = 71.4 tok/s - the slots take turns,
so two sessions share one stream's speed (batched decode would be stage 2). tsvc2 on the service (`logs/tsvc2-seqpick.log`):
every needle found (8K 1484 / 1786, 16K 2510, 31K 3141 tok/s), svcbench 102.4 / 95.1, essay 76.6 / 72.7 tok/s, the tool call,
the picture and its follow-up right, hidden RAM 34 GiB - within today's single-slot spread (8K 1452-1465 / 1873-1881,
svcbench 104-108 / 88-101, essay 80-82 / 74-76).

**1M in either slot** (`r1m.sh`, 18:41-19:01): with both slots' card-0 state allocated at load, a 131K needle in one slot
then a 984K needle in the other (970K tokens, 345 s), its 400-token decode at 70.1 tok/s and a 12K turn (11.6 s) - all
right, no fault. With card 0's state mapped lazily the same sequence garbled the 970K needle's decode (its first token
right, then nonsense, DSpark at 1.08 tokens a pass) and the next request faulted every card (one fault: card 0 reading
one of its expert-cache slabs freed when the cache was released for the prompt) - so the service allocates at load; a
600K needle into a never-used slot passed either way. Not understood yet.

**Limits now.** The slots take turns (two long answers ~36 tok/s each). A prompt being read pauses the other session
for its whole layer-major batch (up to 16K tokens, ~6 s at 16K: windows run only between batches). `BYIELD` is not
acted on. Pictures are read alone. A third request waits for a free slot.

## Measured (2026-10-27 17:00 - 17:30: concurrent sessions, stage 0 - the parked conversations become a pool, two resume bugs fixed)

**Concurrent sessions are the next feature** (the owner, 2026-10-27; the plan: two sessions of up to 1M each, resident on
the cards and taking turns pass by pass, then a measured decision on batched decode). Why not before: Project Maya's
`--parallel` exists only for its GLM engine - its DeepSeek engine answers one request at a time and keeps the others in
RAM / disk slots - and upstream Xolotl's ds41 batch slots (NektoDron, 2026-10-20, after this fork) run on the one-card
ggml path, while larry decodes on its own four-card TP core, whose state and recorded graphs are one sequence's. The
server already has Strata's batch-slot client (`generate_batched`), dormant until an engine announces `batch_slots`.

Stage 0, **in the service since 17:16** (`patches/patch-parkpool.py`, `patches/svc-parkpool.py`, backup
`ds41-xe-engine.sh.pre-parkpool-20261027`):

* **Two resume bugs fixed**, as upstream Xolotl's dffa55a3 found them on its side:
  - a *partial* unpark (`STRATA_SERVE_PARK_PARTIAL`: a prompt that continues most of a parked conversation) resumed at
    any point from 3/4 of it with the side request's window ring and compressor state still live. That is exact only
    within the ring: an even resume point at most 16 positions before the park's end, after `restore()`. Now exactly
    that; otherwise the prompt is read fresh. engine.log had no partial unpark in its 2,014 requests, so no answer is
    known to have been affected.
  - the Engram history (kept by position, not parked) still held the side request's tokens before the resume point,
    which the first new positions' hashes read. Every resume now writes the 8 tokens before it back (the picture
    path did, for pictures only).
* **Taking a parked conversation back no longer throws the live one away**: it is parked first (`park_live`). Before,
  rotating three conversations re-read one of them every time.
* **A pool:** `STRATA_SERVE_PARK_SLOTS` up to 64 (was 2) and `STRATA_SERVE_PARK_GIB`, their host memory together (the
  oldest go first). A park now copies positions / 2 + 2 rows of a ratio-2 source (it copied positions + 1): ~3,200 bytes
  a position instead of 5,120, a 100K-token conversation ~0.33 GB. The service: 8 slots, 4 GiB. The engine prints the
  server's "conversation cache:" lines and INFO `conversation_cache_mib` / `_slots`, so the Monitor's Conversation
  cache card shows the pool.

`bench/conc/s0test.sh` (a private server, 8 slots / 4 GiB; `bench/conc/reusetest.py`):

| | read | reused | time | answer |
| --- | ---: | ---: | ---: | --- |
| turn 1 of four conversations A-D (~3-4K tokens each, one passphrase each) | 3,094-4,071 | 0 | 3.5-4.6 s | right |
| turn 2 of each, in turn: taken back from the pool | 23-24 | 3,094-4,071 | **0.4-0.5 s** | right |
| E after an unrelated side request, its last words re-rendered (the partial resume) | 9 | 2,560 | 0.2 s | right |
| E with a change ~200 tokens before the end (read fresh now) | 2,737 | 0 | 3.2 s | right |

tsvc2 on the service (`logs/tsvc2-parkpool.log`): every needle found (8K 1443 / 1895, 16K 2443, 31K 3158 tok/s),
svcbench 103.2 / 102.0, essay 80.5 / 71.1 tok/s, the tool call, the picture and its follow-up right, hidden RAM 34 GiB at
most.

## Measured (2026-10-26 22:40 - 23:45: pictures - DeepSeek-V4.1's vision path on the Intel engine)

**What was there.** The sidecar GGUF (`V41-MTP-Vision-1017.Q2_K.gguf`) carries the vision tower: 266 tensors, 485M
parameters (a 1024-wide ViT, the aligner up to the model's 5,120, the learned image start / newline / end vectors).
Xolotl's `tools/v41/vision/ds41_vision.py` runs DeepSeek's reference `vision.py` / `image_processor.py` on it behind
strata-vision's line protocol; here on the CPU (tvenv's torch 2.14): it loads in 6.8 s, and an 800x600 picture becomes a
317-token span in 5.7 s on 6 threads. Strata's server already drives such an encoder (its `Vision` class, `GENI` lines
to the engine), but the ds41 engine had no image path and `serve_ds41.py` dropped a message's picture parts.

**The model's own rules** (DeepSeek's `model.py` for 1017): every position of an image span carries the image token
id (129264, `<｜deepseek_image｜>`) and takes the aligner's row (or a learned delimiter vector); positions stay plain
1-D; the router selects with `exp_probs_b_vl` at image positions (our GGUF keeps it for all 40 layers); the Engram
treats them as DEAD - no n-gram crosses one - and adds nothing there.

**`patches/patch-vision-engine.py`**: `GENI` (and `--vision`); the encoder's rows over the token table's at those
positions (`upload_inputs`, f16 like the table); the VL bias in a span's router (spans with pictures take the
composite router - the fused kernel has one bias); the Engram gate 0 and DEAD in its history; the conversation cache
compares the pictures' keys (every image position has the same token id) and ends a prefix match at the first span
that differs; new tokens with a picture run as a batch, their DSpark tail taken from it.
**`patches/patch-vision-server.py`**: Strata's image expansion without Qwen's `<|vision_start|>` pair for DeepSeek's
placeholder; the picture parts reach DeepSeek's encoder; `images_of` in the order the encoder writes the placeholders
(parallel tool results by call id). **`ds41-vision.sh`** (deploy/): the encoder for the config's `"vision"` section.

**Test** (`bench/vision/vtest.sh` / `vtest.py`, two 800x600 cards, a test server): card 1 - *"I see a red rectangle, a
blue oval, and a green downward-pointing triangle. The text reads "LARRY 42"."* (OpenAI API, 9.9 s with the encode);
the follow-up *"What number is written there?"* - *"42"* in 0.2 s (373 of 382 tokens reused, the picture included);
card 2 with the same question - *"a large yellow circle on the left, a purple rectangle on the right, and the text
"BLUE MOON""*, nothing reused (another picture); card 1 through the Anthropic API right (1.8 s: the encoder's cache);
a text-only question as before.

**In the service since 23:16** (`larry-ds41.json`: the `"vision"` section + `--vision`; backup
`larry-ds41.json.pre-vision-20261026`): tsvc2 (`logs/tsvc2-vision.log`) every needle found (8K 1,422 / 1,899, 16K
2,506, 31K 3,185 tok/s), svcbench 105.2 / 102.4, essay 80.6 / 78.0, the tool call right, hidden RAM 34 GiB; the same
picture checks through the service: card 1 7.9 s with its encode (prompt 1.55 s), the follow-up 0.2 s (393 of 403
reused), card 2 right, Anthropic 1.8 s. The encoder holds 2.2 GB of RAM (26 GiB stayed available).

**For a fresh install (2026-10-27 01:10 - 01:25).** The kit's config template turned pictures on, but nothing in the
kit installed what the encoder needs, so a server set up from the repository would not have started. Now
`deploy/bootstrap.sh` makes the torch venv (`tvenv`: torch 2.14.1+cpu, numpy 2.5.4, pillow 13.0.0; 1.0 GB; `VISION=0`
skips it), `deploy/dl.sh vision` fetches DeepSeek's `inference/vision.py` + `image_processor.py` from
deepseek-ai/DeepSeek-V4.1-Flash-1017 at `1bae0e10` (the same blobs as `qv/ds1017`'s copies) into `models/ds1017`, and
`ds41-vision.sh` reads them there - larry's install too (backups `dl.sh` / `ds41-vision.sh` / `tsvc2.sh`
`.pre-kit-20261027`; the running encoder untouched). Tested: `bootstrap.sh` with `BUILD=0` into a scratch `DS41_HOME`
(25 s, the checkout already patched), `dl.sh vision`, then `bench/vision/venc_test.py` (now started through
`ds41-vision.sh`, as the server starts it) through that install: ready in 10.2 s, 317 rows in 5.8 s, row statistics
the same as larry's own install gives (also as the service user). tsvc2 ends with a picture now
(`checks/imagetest.py`: the card, then *"What number is written there?"*): through the service 2.3 s, then *"42"* in
1.2 s. `tools/v41/serve/test_serve_ds41.py` gained 8 picture tests on a stub encoder (each placeholder's span in its
place, parallel tool results' pictures in call order, a typed `<｜deepseek_image｜>` kept as text, one key per
picture): 50 pass.

## Measured (2026-10-26 19:50 - 20:15: the fused hc path back on - its 1M failure was the kept batch's VRAM edge)

**`STRATA_HC_FUSE_MAXN=16384` again** (`patches/svc-hcon.py`, backup `ds41-xe-engine.sh.pre-hcon-20261026`): it was off
since 12:50 because abwT131's 973K turn died in `DSV41_HC_FA` with `UR_RESULT_ERROR_OUT_OF_RESOURCES` - but abwT141's
turn died the same way with it off, and the VRAM trace showed why (the section below). abwT144, two fresh servers
each, interleaved, the depth allowance (2.0) in both:

| | 8K first / again | 16K first / again | 31K first / again | 984K needle | its 11.7K turn |
| --- | --- | --- | --- | --- | --- |
| off (8) | 5.94, 5.90 / 4.53, 4.53 s | 6.70, 6.59 / 6.48, 6.44 s | 10.78, 10.72 / 10.94, 10.96 s | 357-358 s (abwT141-143) | - |
| on (16384) | 5.91, 5.91 / 4.57, 4.47 s | **6.43, 6.43 / 6.24, 6.25 s** | **10.33, 10.29 / 10.67, 10.50 s** | **345.7 s**, found | 14.4 s, 1.66 GiB free at its low point |

-3 to -4 % on prompts from 16K up, 8K even; the 1M turn passed with room to spare.

**The agent replay on the final config** (abwT145, `agentx.py`: a coding session growing to ~43K tokens, 18 follow-ups
of 17-6,500 new tokens and one 22K side request; two fresh servers each): follow-ups 38.90 / 38.81 s summed (this
morning, abwT112: 39.9), all prompts 59.5 s, decode 127 tok/s (122.9). With the Engram drop off: follow-ups 39.77 /
38.78 s - the same - but one server's 22K side request stalled (16.3 s against 8.6): the drop costs this workload
nothing.

**What bounds an agent turn now** (the same replay's engine log): a follow-up's expert phase is ~1.4-2.2 s whether it
has 660 or 3,800 new tokens, its dense phase 0.2-1.3 s. A few hundred tokens already touch nearly every expert of
every layer (top-6 of 384), so the ~40 % that live in the RAM arena (80.3 GiB, 7,753 experts, 10.6 MiB each) cross
PCIe or run on the CPU pool every turn: a 3,618-token turn 14,855 VRAM hits / 319 CPU / **6,713 streamed** (~70 GiB,
3.66 s), a 660-token turn 11,755 / **1,646 CPU** / 1,283 streamed (1.86 s). The four cards share one PCIe Gen4 x16
uplink (~31.5 GB/s) and the CPU pool is DRAM-bound (the 5800X's two DDR4 channels): mid-size turns sit at the box's
memory bandwidth. What would move it: fewer bytes per arena expert (a lower-bit quant of the cold experts - a quality
trade) or more VRAM; the exact engine has little left there.

## Measured (2026-10-26 18:50 - 19:50: a 1M conversation's follow-up turn could run card 0 out of memory - more room the deeper the conversation)

**Long contexts after the page-cache changes** (abwT141, one fresh server): the 131K needle 41.2 s (41.1 before), its
turn 6.0 s; the 984K needle 357.0 s (356.6), found - but its 11,684-token follow-up turn **died**:
`UR_RESULT_ERROR_OUT_OF_RESOURCES` in `DSV4_HC_PRE` on card 0. The same turn had passed in abwT132 (10.0 s).
`STRATA_VRAM_TRACE=1` (abwT142, twice, both passed: 10.0 / 14.1 s): the keep-cache shrink left 10.27 GiB free on card 0
and the 11.7K batch took all of it - **0.00 GiB free at its low point**; abwT141's shrink had left 9.85. Neither
estimate covers a batch at 1M: the calibrated one (3.5 GiB + 0.45 MiB a column + the margin) nor the learned one (the
last kept batches, depth-scaled); a bigger learned margin (`STRATA_SERVE_KEEP_LEARN_MARGIN_GIB=2.5`) changed nothing,
the calibrated path sized that shrink. This morning's "the fused hc path dies at 1M" (abwT131, `DSV41_HC_FA`) was
most likely the same edge, not the fused kernel.

**`STRATA_SERVE_KEEP_LEARN_DEPTH_GIB`** (`patches/patch-keepdepth.py`, engine default 0; the service 2.0,
`patches/svc-keepdepth.py`, backup `ds41-xe-engine.sh.pre-keepdepth-20261026`): a kept-cache batch gets g * min(1,
tokens / 1M) GiB more room on card 0 - in the shrink (both estimates) and in the batch width. abwT143 (1.5): the
shrink left 11.28 GiB, the batch took 10.54, **0.73 GiB free at its low point**, the turn 15.0 s. The service runs 2.0
(~1.2 GiB to spare at 1M; +0.06 GiB on a 31K conversation). A 1M turn varies 10.0-15.0 s run to run (its experts phase
2.6-8.1 s).

## Measured (2026-10-26 17:50 - 19:00: the stalls without touching swap - the load and the Engram reads stop flooding the page cache)

**Where the pressure comes from** (`bench/speed/memtrace3.bt`: page-cache insertions by file, anonymous faults,
swap-ins, TTM backups; `memsum3.py`; abwT138, two fresh servers as the service runs them). At every start the engine put
**182-186 GB** into the page cache, nearly all of it the model GGUF: the tier's bulk reads (`read_all`: the experts it
places in VRAM and the RAM arena, the replicas) used `Ds41FileBlobs::read`, a buffered pread - only the prefetch and the
split reads used the source's O_DIRECT twin. On a box the engine fills, that flood pushed the arena's own pages to swap
(m2's first 8K swapped 7.2 GB back in: 17.0 s) and had the xe shrinker back up 49 GB of the engine's buffers in one
load. During the prompts the Engram reads added 12-14 GB more (8K / 16K / 31K twice); anonymous faults ~2 GB.

**`STRATA_XE_LOAD_DIRECT`** (`patches/patch-loaddirect.py`, default 1): `read_all` reads through `read_staged`
(O_DIRECT, a per-thread aligned scratch; falls back to the buffered read on any short read). abwT139 (three fresh
servers each): page-cache insertions while loading 156-198 GB -> 2-6 GB, the load as fast (80-86 s); the buffered arm's
o1 still swapped in during its first prompts (8K 9.28 s, 31K 13.39 s), and the direct arm's d3 still had 1.8 GB backed
up during its prompts (8K 7.83 s) - the Engram reads.

**`STRATA_ENGRAM_DROP=1`** (`patches/patch-engdrop.py`, engine default 0; in the service, `patches/svc-engdrop.py`,
backup `ds41-xe-engine.sh.pre-engdrop-20261026`): a prompt batch (> 64 tokens) drops the Engram weight pages it read
(`posix_fadvise` DONTNEED; a weight page holds 15-16 random rows, the row cache keeps the rows; the prefetch's pages stay
until the batch reads them; the scale pages - 8 bytes a row, shared by 512 rows - stay cached). abwT140, the load
direct in both arms:

| | first 8K | first 16K | first 31K | 31K again | TTM backups during the prompts | direct reclaim scans |
| --- | --- | --- | --- | --- | --- | --- |
| off (k1-k3) | 6.89 / 6.17 / 5.97 s | 6.68 / **8.48** / 6.68 s | 10.68 / **17.86** / 10.87 s | 10.60 s | 0 / **2.8 GB** / 21 MB | 3.1 / 27.4 / 1.3 GB |
| on (r1-r3) | 6.97 / 7.26 / 6.19 s | 6.79 / 6.71 / 6.77 s | 10.73 / 10.78 / 10.74 s | 11.02 s | 26 / 29 / 0 MB | 1.0 / 1.0 / 0 GB |

No stall with it; the price is a repeated big prompt's Engram rows (~4 % on the second 31K - its rows no longer page
cache hits). With both (abwT140's "on" arm) the worst first prompt ran ~1 s over its usual (r2's 8K, 7.26 s), against
+7 s (k2's 31K) without the drop and +10 s (m2's 8K) with neither. Swap off would still remove the mechanism itself
(the owner's decision); these two remove most of what set it off.

**Test harness:** the service's user cannot open `/dev/dm-0`, so the service reads layer 14's Engram table through
ext4, while every A/B test server (root) before abwT138 took the raw-device path (`RawMap`). `tabx.sh` now starts its
servers with `STRATA_ENGRAM_RAWDEV=0` (a config can still set it).

## Measured (2026-10-26 17:10 - 17:50: Project Maya's DSpark cut in the service; the server on DeepSeek's edge cases)

**`STRATA_DSPARK_CUM=0.5` in the service** (`patches/patch-dsparkcum.py`, engine default 0 = off;
`patches/svc-dsparkcum.py`, backup `ds41-xe-engine.sh.pre-dsparkcum-20261026`). Besides the 0.3 floor on each draft's
own confidence, the verify window stops where the product of the drafts' confidences so far drops below 0.5 - a window
whose later drafts are likely lost is not verified whole (Project Maya's `MAYA_DS4_DSPARK_CUM`; lossless, only the
verify width changes). abwT137, three fresh servers each, interleaved, the decode tasks (400 tokens each):

| | essay | code | math | all (tokens / time) |
| --- | ---: | ---: | ---: | ---: |
| off | 81.2 tok/s (2.31 tok/pass, 28.4 ms) | 112.8 (3.29, 29.1) | 124.1 (3.87, 31.1) | 102.7 tok/s |
| 0.5 | 82.8 (1.78, 21.5) | **118.7** (2.87, 24.1) | 128.6 (3.63, 28.2) | **106.2** |

tsvc2 (`logs/tsvc2-dsparkcum.log`): every needle found, svcbench 94.3 / 98.0, essay 81.5 / 74.5, the tool call right.
That run's first 8K needle read at 516 tok/s (15.5 s): the swap stall again, right after a start that followed a
detour through the small layout (19 GiB in swap).

**The server on DeepSeek's edge cases** (`patches/patch-ds41conf.py`; `tools/v41/serve/test_ds41_conformance.py`, the
cases of Project Maya's conformance tests): three bugs fixed - (1) a `developer` message (OpenAI's newer system role)
failed the request ("Unknown role: developer"), now a system turn; (2) Anthropic parallel tool results stayed in the
order the client sent them: the adapter (`serve/frontend.py`) dropped `tool_use.id` / `tool_result.tool_use_id`, so
DeepSeek's encoder could not sort them into call order, and DSML results carry no ids - the model paired results with
the wrong calls; the ids are kept now; (3) a DSML tag inside a parameter value (a file the model writes about DSML) broke
the call: the `<｜DSML｜ calls>` wrappers were dropped from the whole stream and a value's `</｜DSML｜ invoke>` ended
the call; values are skipped as DeepSeek's parser does (only `</｜DSML｜ parameter>` ends one). Two differences stay,
reported as notes: newlines right after `</think>` are dropped (the base server's Qwen habit) and a long string
argument arrives whole (the keep-alives hold idle clients). Tests: test_ds41_conformance all passed, test_serve_ds41 42,
the base server's test_server 314 (one expectation updated for the kept id), test_frontend 25, test_env_to_user 6.

**A restart while the engine loads fails over to the small layout**: `larry-mode restart` ~45 s after a window's
restore (health 200 comes from the server before the engine is ready) - "the server did not exit", then the small
layout (vllm-a) came up; `larry-mode ds41` after it finished brought the service back. Restart only after the engine
has loaded (larry-mode's "loaded in").

## Measured (2026-10-26 16:40 - 17:10: swap off ends the first-prompt stalls; message text can no longer open turns)

**Swap off** (`bench/speed/swapctl.sh`, abwT136: three fresh servers with swap off, three with it on, interleaved,
`memtrace2.bt`; `swapctl.sh guard` turned swap back on after the window):

| Swap | First 8K | First 16K | First 31K | TTM backups during the prompts | Rebind worker (longest) | Engine direct reclaim (longest) |
| --- | --- | --- | --- | --- | --- | --- |
| off (s1-s3) | 6.59 / 6.14 / 6.27 s | 7.14 / 6.62 / 6.67 s | 10.63 / 10.62 / 10.61 s | none | 85-110 ms (22 ms) | 9-26 s summed (61 ms) |
| on (n1-n3) | 6.38 / 7.49 / 6.05 s | 6.64 / 6.58 / 6.61 s | **16.43** / 10.96 / 10.61 s | 1.25 / 0.04 / 0 GiB | 2.9 s in n1 (2.5 s) | up to 458 s summed (641 ms) |

With no swap TTM has no backup space, so the xe shrinker writes nothing out: no backups, no restores, no long
rebinds, no swap-ins, and an eighth of the time in direct reclaim. Over abwT134-136, 5 of 12 swap-on fresh servers
had a first prompt 1.5-20 s slower; none of the 3 swap-off ones. (The first 8K's remaining ~2 s over the second are the
first calls' own costs - oneMKL kernels, oneDNN partitions, buffer exports - not memory.) To adopt it for the service
only: a systemd drop-in that runs `swapoff /swapfile` before the start and `swapon /swapfile` after the stop (other
layouts keep their swap) - not installed: the owner's decision. **Decided 2026-10-26 ~20:40: swap stays on** - the
engine-side fixes (17:50-19:00: the load through O_DIRECT, the Engram drop) cover the stalls.

**Message text cannot open turns any more** (`patches/patch-literals.py`, in `tools/v41/serve/serve_ds41.py`; live since
the window's restart). V4.1's `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`, `｜DSML｜` and its tool / fim markers are
USER_DEFINED tokens (type 4), which Strata's tokenizer matches in any text, and the base server escaped only the CONTROL
ones (type 3) and `<think>`: a file an agent read or a tool result holding one became a real role or tool-call token.
The server now marks every special as text inside messages, tool results and tool-call arguments (Project Maya's
DeepSeek server does the same). `test_serve_ds41.py` checks it (fails on the old code) and now runs its HTTP test under
the venv that runs the tests.

**Project Maya** (github.com/mw00/project-maya, MIT, the same Strata lineage; its `maya-ds4` engine runs this model on
CUDA / ROCm), reviewed for what to take: its exact prompt skip (`MAYA_DS4_PROMPT_SKIP`) is what our `--enc-skip`
already does (on in the service: layers 21-39 run only on a prompt's last 2,560 tokens); its two-plus parked
conversations we have too (`STRATA_SERVE_PARK_SLOTS`, in RAM; Maya also keeps them on disk across restarts). Taken: the
special-token escaping above, and its DSpark cut on the product of the drafts' confidences (next section).

## Measured (2026-10-26 15:54 - 16:30: the kernel's memory settings - worse; the first-prompt stalls are the xe shrinker writing the engine's buffers out)

**The three settings the owner approved trying** (`bench/speed/memset.sh tuned`: `vm.watermark_scale_factor` 10 -> 200,
`vm.compaction_proactiveness` 20 -> 0, THP defrag `madvise` -> `defer`; runtime only) against the boot values, abwT134:
four fresh servers each, interleaved, fresh 8K / 16K / 31K prompts twice each, with a kernel trace over the whole window
(`bench/speed/memtrace.bt`, bpftrace: direct reclaim and direct compaction per thread class, page migrations, xe
userptr invalidations, exec ioctls; `vmwatch2.py`: /proc/vmstat; `memsum.py` sums both per config):

| Settings | First 8K (each server) | 8K again | 16K first / again | 31K first / again | Engine direct reclaim during the prompts | Swap-ins |
| --- | --- | ---: | --- | --- | --- | ---: |
| boot values | 15.69 / 6.35 / 6.22 / 6.48 s | 4.55 s | 6.62 / 6.35 s | 10.56 / 10.56 s | 2.7K calls, 5.4 s (summed over threads) | 4.2K pages |
| tuned | 26.87 / 7.26 / 6.30 / 6.17 s | 4.60 s | 6.67 / 6.75 s | 10.81 / 10.66 s | 11.4K calls, 336 s | 155K pages |

Worse: not adopted, and the boot values are back (nothing was persisted). The higher watermarks keep 7-11 GiB free
(the boot values ~1 GiB) on a box whose working set already fills its RAM, so the kernel reclaims and swaps more:
kswapd's scans x2.2, the engine's time in direct reclaim x60, 620K pages swapped in during t1's first prompt.
Compaction was not the cause: no userptr invalidations at all, no exec ioctl over 2 ms; proactiveness 0 cut the page
migrations 4x with no effect on the stalls; the engine has no THP regions, so `defer` changes nothing for it.

**The stalls are the xe shrinker writing the engine's own buffers out.** The stacks behind the direct reclaims and
compactions led into TTM: `ttm_tt_restore` under `xe_bo_cpu_fault` and under the xe preempt-rebind worker
(`preempt_rebind_work_func`). `bench/speed/memtrace2.bt` adds TTM's backups and restores and the rebind worker
(abwT135, two fresh servers on the boot values). **e1 stalled** (16K 8.07 s against 6.37 again, 31K 12.71 against
10.59): during its prompts the shrinker backed up 457K pages (1.75 GiB) of the engine's system-memory BOs, 40 restores
read them back (787 ms, the longest 546 ms), and the rebind worker - a compute VM's queues stay preempted while it runs -
ran 293 times for 4.96 s (the longest 1.23 s). **e2 did not stall**: no backup, no restore, 88 ms of rebinds. TTM backs a
BO up to shmem only while swap has room (`ttm_backup_bytes_avail` returns `nr_swap_pages` in bytes - checked in larry's
ttm.ko), so with larry's 128 GiB swapfile the shrinker may move any unpinned BO out under pressure. The pressure during
prompts comes from buffered reads (the Engram tables: 256 reader threads, and layer 14's `RawMap` on /dev/dm-0 without
O_DIRECT) and anonymous faults: up to ~250 engine threads in direct reclaim at once (t1), each running the shrinkers.

**Options, none done yet:** no swap (`swapoff`: TTM then has no backup space and the shrinker backs nothing up - the
owner's call, other layouts may count on the swapfile); fewer Engram readers (Project Maya's ds4 engine uses
min(16, max(4, cores / 2)) buffered ones) or O_DIRECT for the RawMap; `posix_fadvise(DONTNEED)` on the model file's
ranges as the load copies them (~150 GiB goes through the page cache at every start); `mlock` of the pageable arena.

## Measured (2026-10-26 07:55 - 10:27: card 0 still bounds - wo_a as one product, the hc mixes in one pass)

**In the service since 10:23** (`patches/svc-oproja.py`, backup `ds41-xe-engine.sh.pre-oproja-20261026`; binaries built
10:21): `STRATA_OPROJ_A=1` + `STRATA_HC_NORM_MV_WIDE=1`. tsvc2 (`logs/tsvc2-oproja.log`): every needle found (8K 1139 /
1833, 16K 2354, **31K 2990** tok/s), svcbench 93.3 / 96.4, essay 76.8 / 71.1 tok/s, the tool call right, hidden RAM 33
GiB at most.

**Which side bounds a big prompt now.** 1024-column spans (abwT115) again bought nothing (31K 11.34 -> 11.23 s, 4.5K
+0.34 s), and the server's split moved from "dense" to "experts" as before. Card 0's GPU timeline says why
(`STRATA_GPU_PTL`, abwT114 at 512 columns, abwT117 at 1024): its queue is 94-96 % busy through layers 0-19 at both widths,
13.1 ms a 1024-column span = 6.54 ms a 512 one against 6.82. The server's dense / experts split is host time, not the
cards': card 0's GPU work is the bound (~230 ms a layer of a 16K batch, the expert cards' calls ~165-175 ms). The expert
GEMMs alone (`bench/dqb/bench_dqgemm`, abwT116; `bench/dqb/dqb-shapes.sh` sources oneAPI first): xe_dqgemm tops out near
50 TFLOPS at 128+ tokens an expert (Q3_K 46, Q2_0 50, PTQ1_0 48; 24-36 at 32-64) - its A fragments come from local memory
one DPAS each - so the tier has room, but card 0 is first.

**Card 0's span by op** (`bench/speed/opprof-shapes.sh`: `GGML_SYCL_OP_PROF_SHAPES=1`, `patches/patch-opprof5.py` - each
label gets the node's shape and its first source's name / type; abwT118, p30k's second batch, a layer-span = one layer's
512 columns):

| a layer-span (abwT118) | us | |
| --- | ---: | --- |
| SDPA: oneDNN execute + its Q' / mask kernels + the op's other kernels | 1123 + 232 + 299 | the attention |
| DSV41_KGATHER (each query's 656 key rows gathered, 387 MB written) | 784 | |
| wo_a (`larry_slices_gemm`: 8 slices, each gathered to fp16 + its own GEMM) + the CONT after it | 768 + 162 | 277 us of host a call |
| q_b / wo_b | 446 / 419 | 115 / 102 TFLOPS |
| the hc mixes: rms_norm [20480 x 512] + MUL_MAT [24 x 512], twice | 2 x (197 + 126) | |
| HC_POST, HC_PRE, the state / output copies | ~2 x 173, ~2 x 120, ~0.4 ms | |

* **`STRATA_OPROJ_A=1`** (`patches/patch-oproja.py` + `patch-oproja2.py`; ggml.h / ggml.c `ggml_dsv41_oproj_a` = a
  `GGML_OP_DSV41_OPROJ` node with op_params[0] = 1): wo_a as one product written in wo_b's input layout - o converted to
  fp16 once, the per-slice planar slices dequantised into one fp16 cache entry, the 8 slices in ONE oneMKL gemm_batch.
  oneMKL's strided batch refuses interleaved outputs ("Illegal value supplied for parameter stride_c": abwT121's runs
  aborted, rc 134), so the pointer form (one group of 8, the device pointer arrays written by a kernel). 768 + 162 ->
  **520 us** (abwT122). The strided form into the old layout (`GGML_SYCL_LARRY_SLICES_BATCH=1`, `patch-slicesbatch.py`,
  default 0) gave 490 us + the CONT (abwT120).
* **`STRATA_HC_NORM_MV_WIDE=1`** (`patches/patch-normmvw.py`): the prompt path's hc mixes through
  `ggml_dsv41_norm_mv` (the norm folded out of the product) with a wide kernel - a work-group = 64 tokens x a 640-value
  slice of the 20480, the slice's weight rows staged in local memory once, a sub-group = 2 tokens, partials then one
  pass. 2 x 323 -> **2 x 153 us**. Forms that lost: no staging (the weight through L2 for every token: 349 us), 16
  tokens a work-group with scalar staging (120 dependent loads a work-item: 604 us; abwT120's prompts 0.1-0.2 s slower).

Numerics of both (`bench/speed/tlppl-ab.sh opb`, abwT123; the script runs any switch): p8k KL mean 2.62e-3 against a
floor of 2.54e-3 (p99 3.91e-2 / 3.80e-2, max 0.31 / 0.36), p30k PPL 1.7833 / 1.7829. (abwT120's floor run had one
token at KL 12.4 with no change at all: the single big-KL tokens are run-to-run, not a kernel.)

| abwT123 (server, fresh) | without | with both |
| --- | ---: | ---: |
| 8K (second run) | 4.93 s | **4.51 s** |
| 16K | 7.23 / 6.74 s | 6.61 / **6.40 s** |
| 31K | 11.71 / 11.51 s | **10.78 / 10.96 s** |
| agent replay: follow-ups / side prompts | 40.23 / 16.09 s | 39.84 / **15.30 s** |
| decode (agent replay) | 122.0 tok/s | 120.4 tok/s |

Card 0's batch 4753 -> 4131 ms in the profiled range (abwT122; the profiler's skip lands on fewer layer-spans now, so
compare per call). Follow-ups barely move: their layers wait on streamed experts and the CPU pool.

* **`STRATA_HC_FUSE_MAXN=16384`** (in the service since 10:54, `patches/svc-hcwide.py`, backup
  `ds41-xe-engine.sh.pre-hcwide-20261026`; `patches/patch-hcwide.py`): the fused hc ops (hc_fa / hc_fb: hc_post, the mixes'
  dots and hc_pre in one pass, the outputs written in place - `STRATA_DIRECT_OUT`) at prompt widths too. They lost there
  before (31.6K 26.2 -> 27.7 s, 2026-10-22) because hc_fa's sub-group per (token, chunk, row group of 6) re-read the
  token's state for each of 4 row groups and every token read all of fn through L2; its wide form (n > 8,
  `GGML_SYCL_HC_FA_WIDE=0` = the narrow one) takes 16 tokens x a 128-column chunk a work-group with the chunk's fn rows
  staged in local memory, a sub-group = one token with all 24 rows. hc_merge (<= 64 tokens) stays off past 64. In the
  profile the separate HC_POST / HC_PRE / mixes and the state copies go: HC_FA 340 us (~147 MB moved: near bandwidth),
  HC_FB 71, the finish (HC_POST2) 198. Numerics (`tlppl-ab.sh hcw`, abwT124): p8k KL 2.72e-3 against 2.70e-3, p30k PPL
  1.7799 / 1.7825.

| abwT124 (server, fresh; the config of 10:23 without / with it) | without | with |
| --- | ---: | ---: |
| 8K (second run) | 4.79 s | **4.50 s** |
| 16K | 6.80 / 6.45 s | 6.61 / **6.23 s** |
| 31K | 10.90 / 10.96 s | **10.55 / 10.35 s** |
| agent replay: follow-ups / side prompts | 39.64 / 15.31 s | 39.25 / **14.81 s** |
| decode (agent replay) | 122.9 tok/s | 124.0 tok/s |

Over the day (fresh 31K through the server, second runs): 12.34 s (07:30) -> 11.31 (the indexer) -> 10.96 (wo_a, the
mixes) -> 10.35 (the fused hc path, since off: see "984K" below) -> 10.24 (the fp16 attention output). The final config
(12:49, abwT133, two servers): 8K **4.48** / 4.80 s, 16K **6.39** / 6.41 s, 31K **10.53** / 10.59 s (07:30: 4.70, 7.00,
12.34).

* **`STRATA_SDPA_O16=1`** (in the service since 12:11, `patches/svc-sdpao16.py`, backup
  `ds41-xe-engine.sh.pre-sdpao16-20261026`; `patches/patch-sdpao16.py`: ggml.h / ggml.c `ggml_dsv41_attn_sdpa_o16`, the
  same op with an F16 result): the prompt attention's output stays fp16 - the oneDNN path copies O' through, the oneMKL
  path's second gemm_batch writes fp16, the in-place de-RoPE takes F16, `ggml_dsv41_oproj_a` reads it as it is (wo_a
  always read o rounded to fp16). wo_a's op 543 -> 337 us a span, the attention op's own kernels 382 -> 323 us. Numerics
  (abwT129): p8k KL 2.22e-3 against 2.34e-3, p30k PPL 1.7773 / 1.7816. Fresh (two servers each, interleaved): 16K 6.25 /
  6.29 -> 6.17 / 6.16 s, 31K 10.32-10.35 -> 10.16-10.27 s - card 0's ~0.26 ms a span shows up as ~1 %: the expert cards'
  waits grew as much (31K "experts" 1.64-2.19 -> 1.79-2.44 s). Card 0 and the expert cards are balanced now.

**Long contexts on the 12:11 config** (abwT130, `tabx.sh needles`): a 131K needle from empty in **41.1 s** (3,192 tok/s;
Oct 24's build 80 s), found, decode at that depth 85.9 tok/s, an 11.7K-token turn on it 5.9 s; a 256K needle in **73.8 s**
(3,466 tok/s), found, decode 82.2 tok/s, an 11.7K turn 6.2 s.

**984K, and the fused hc path off again.** abwT131: a 973K needle in **348 s** (2,796 tok/s; Oct 24: 1,140 s), found,
decode at that depth 67.9 tok/s - then the 11,684-token turn on it (a 13,139-column batch fit to card 0's 10.27 GiB
free) failed in `DSV41_HC_FA` with `UR_RESULT_ERROR_OUT_OF_RESOURCES` (the server restarted the engine). With
`STRATA_HC_FUSE_MAXN=8` the same run passed (abwT132: needle 356.6 s, found, decode 66.6 tok/s, the turn 10.0 s), the
same batch width and the same first BAR staging on card 0 before it. On Larry this error has meant card 0's VRAM running
out (a 32K batch's GEMM scratch, an under-sized learned batch), so the fused path likely raises card 0's peak at a
1M-context turn, though not at 31K. **Off in the service since 12:49** (backup `ds41-xe-engine.sh.pre-hcoff-20261026`;
the code stays, the default 8). To bring it back: `STRATA_VRAM_TRACE=1` on both paths at 973K, then a larger
`STRATA_SERVE_BATCH_COL_MIB` or a smaller kernel footprint. tsvc2 on the 12:49 config (`logs/tsvc2-hcoff.log`): every
needle found (8K 992 / 1846, 16K 2343, 31K 3017 tok/s; warm repeats 2586 / 3023), svcbench 96.9 / 91.0, essay 78.0 /
73.6 tok/s, the tool call right, hidden RAM 34 GiB.

**The balance after it** (abwT125, `STRATA_GPU_PTL`): card 0's span 6.49 ms (first 16K batch) / 6.05 (second), its idle
before the combines 482 / 411 ms a batch (was 183-346) - it waits on the expert cards more; a layer of a 16K batch ~218
ms of card 0's work against the expert cards' two calls ~176 ms. Not adopted: `STRATA_XE_FN16=1 / 2` again (abwT126:
31K 10.42 / 10.42 / 10.30 s, 16K 6.32 / 6.28 / 6.24 - within the noise).

**Known: a fresh engine's first big prompts stall on host memory.** On 3 of ~14 fresh servers today the first 16K or
31K prompt took 15-22 s (abwT112 s2, abwT117, abwT126 f0: 7-10 s of tier waits), and on all of them the first 8K takes
6.0-8.5 s against ~4.45 for the second. In the traced one (abwT117) card 0 idled 70-200 ms before nearly every combine
while the expert cards' calls themselves ran at normal length: the calls started late, each layer's staging done only
5-10 ms before its call (17-19 ms normally); that run's card-0 cache regrow spent 245 ms on its slabs (50-95 normally).
`bench/speed/vmwatch.sh` (/proc/vmstat every 2 s, abwT127 / abwT128) over a fresh server's first prompts: kswapd
reclaims 0.1-2.7 M pages / 2 s, direct reclaim up to 2 M pages / 2 s, direct compaction stalls up to 330 / 2 s, page
migration up to 470 K pages / 2 s - in every big prompt, worst in the first minutes after the arena loads; between
prompts, none. `STRATA_ENGRAM_DIRECT=1` (the paged Engram table with O_DIRECT) does not change that churn (abwT128: the
same order of reclaim; first prompts 6.0-6.5 vs 6.35-7.2 s, two servers each) - left off. Not tried then (system
settings, the owner's call; tried at 15:54 with the owner's OK - worse, see that section): `vm.watermark_scale_factor` (kswapd earlier, fewer direct reclaims), `vm.compaction_proactiveness=0`,
THP defrag `defer`; and O_DIRECT for layer 14's raw block-device reads (`RawMap` opens /dev/dm-0 without it).

## Measured (2026-10-26 06:35 - 07:51: the prompt indexer as one fused XMX kernel, told each query's visible rows)

**In the service since 07:48** (`patches/svc-lifused.py`, backup `ds41-xe-engine.sh.pre-lifused-20261026`):
`GGML_SYCL_LI_FUSED=1` + `STRATA_LI_NVIS=1`, binaries built 07:08 (`bench/li/lif-build.sh`; the 02:55 ones, with the
dormant `GGML_SYCL_LI_S16`, in `logs/tier-rebuild/bin-20261026-0255`). tsvc2 (`logs/tsvc2-lif.log`): every needle found
(8K 1350 / 1791, 16K 2222, **31K 2751** tok/s), svcbench 98.9 / 94.1, essay 76.0 / 70.4 tok/s, the tool call right,
hidden RAM 33 GiB at most.

**Why**: card 0's op profile of a 31K prompt's second batch (abwT110) had the prompt indexer (`LIGHTNING_INDEXER`, li_gemm)
at 5.7 ms a call, 9.8 % of the card. li_gemm runs one oneMKL fp16 GEMM that writes S = Q [512 queries x 32 heads x 128]
. K^T in f32 - at 16K keys 1 GB a call - and a second kernel that reads S back for the relu-weighted head sum; the
GEMM's own arithmetic is ~0.4 ms. Three of a span's layers run it (90 calls in that batch), on cap keys = the next power
of two of the span's visible rows.

**abwT111, `GGML_SYCL_LI_S16=1`** (S in fp16: the code the 02:55 build already carried, off): card 0's indexer 5.70 ->
3.53 ms a call, so the bytes were the cost; fresh prompts (second runs) 16K 7.07 -> 6.92 s, 31K 12.44 -> 12.24 s; p8k KL
mean 3.73e-3 against a floor of 3.06e-3 (one token at 6.8), p30k PPL 1.7783 against 1.7833. Not deployed: the fused
kernel below needs no S at all.

**`GGML_SYCL_LI_FUSED=1`** (`patches/patch-lifused.py`, lightning-indexer.cpp): one XMX kernel, S never leaves the
registers. A sub-group = one query's 32 heads (4 row tiles of 8) x 32 keys (2 column tiles of 16), K = 128 in 8 steps,
fp16 x fp16 -> f32 as li_gemm's GEMM; B = the keys transposed, loaded `col_major` straight from the (fp16-converted) key
rows; an accumulator lane holds its key's column (element i = head row i, `bench/dqb/coord_probe.cpp`), so the head sum
is lane-local; 32-key steps the query cannot see (all -inf) write -inf without the multiplies. `bench/li/li_probe.cpp`
(card 1, 512 queries, the same inputs through each):

| keys (cap : visible rows) | li_gemm | S in fp16 | fused | max \|diff\| / max \|ref\| (fp16 S / fused) |
| --- | ---: | ---: | ---: | --- |
| 4K : 3.9K | 1.11 ms | 0.59 ms | 0.19 ms | 2.7e-4 / 2.1e-7 |
| 16K : 8.4K | 4.34 | 2.29 | 0.44 | 3.2e-4 / 2.0e-7 |
| 16K : 15.8K | 4.36 | 2.30 | **0.63** | 2.6e-4 / 1.8e-7 |
| 32K : 31.6K | 8.71 | 4.58 | **1.22** | 2.5e-4 / 1.9e-7 |
| 64K : 60K | 17.4 | 9.16 | **2.33** | 2.8e-4 / 1.9e-7 |
| 256K : 250K | 69.4 | 36.6 | **11.2** | 3.0e-4 / 2.7e-7 |

~110 TFLOPS at 32-64K keys (fp16 XMX peak ~175). The forms tried (`li_probe`'s f2 / f4 / f2q, B `col_major` or a VNNI
repack): 64 keys a step (16 accumulators, GRF256) and the query's 32 A tiles kept in registers (GRF256) lost to the
32-key step re-loading A from L1 (GRF128) everywhere past 16K keys; the VNNI repack's scattered 2-byte stores cost up to
6 ms at 256K keys. Work-group: 8 queries x 128 keys (256 to 64K keys, 512 beyond; 16 queries past 128K) - the sweep's
best, `GGML_SYCL_LI_FUSED_QW` / `_KR` override. `GGML_SYCL_LI_FUSED_CHECK=N` runs the first N calls through li_gemm into
scratch as well: in the engine (p8k, p30k) max |diff| 1e-6 of scores up to 5.6, 0 non-finite mismatches.

**`STRATA_LI_NVIS=1`** (`patches/patch-linvis.py`: ggml.h / ggml.c `ggml_lightning_indexer_nvis`, ggml-sycl's supports_op,
li_fused and li_gemm, ds41_dense.cpp): the prompt path built a [cap, n] 0 / -inf mask per call (repeat + add + step + log
+ an F16 copy: ~0.9 ms a call at 16K keys) and read it again for the selected rows' mask; now the op takes each query's
visible row count (op_params[0] = 1, src[3] = nvis F32 [n]; the CPU op asserts on it) and the selected rows' mask is
log(step(nvis - sel)) on [K, n].

| p8k (`bench/speed/tlppl-lif.sh`, KL against run a) | PPL | KL mean | p99 | max | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| a / b (without) | 1.6809 / 1.6765 | - / 2.74e-3 | - / 3.82e-2 | - / 0.51 | - / 98.81 % |
| f (fused) | 1.6776 | 3.40e-3 | 3.75e-2 | 6.79 | 98.78 % |
| n (fused + nvis) | 1.6774 | 2.66e-3 | 4.06e-2 | 0.46 | 98.83 % |

The fused run's mean is its one outlier token (6.79 / 8538 rows = 8e-4 of it; the fp16-S run in abwT111 had a 6.8 one
too, the nvis run with the same kernel none). p30k PPL: without 1.7816, fused 1.7823, both 1.7812.

| card 0, p30k's second batch (`bench/speed/opprof.sh`) | abwT110 | S in fp16 | fused + nvis |
| --- | ---: | ---: | ---: |
| card 0's time | 5218 ms | 5065 ms | **4783 ms** |
| `LIGHTNING_INDEXER` a call (90 calls) | 5699 us (host 162) | 3531 us | **684 us** (host 24) |
| the mask: REPEAT + STEP + LOG + its F16 CPY | 76 ms | 76 ms | 6 ms |
| (idle before a graph) | 658 ms | - | 782 ms |
| the one-shot prompt (`--prompt-batch 16384`) | 14.94 s | 14.40 s | 13.11 s |

| abwT112 (server, fresh, two runs each) | without | fused | fused + nvis |
| --- | ---: | ---: | ---: |
| 8K | 6.75 / 4.70 s | 7.11 / 4.63 s | 6.28 / 4.62 s |
| 16K | 7.33 / 7.00 s | 7.02 / 6.80 s | 7.48 / 6.76 s |
| 31K | 12.44 / 12.34 s | 11.82 / 11.47 s | 22.04 / **11.31 s** |

The 22.04 s: 9.97 s of tier waits in that one batch, nothing in its log (no stream-buffer cap - all three servers capped
21 times, during the first 8K prompts - no stall watch, nothing in the kernel log); the next 31K on that server 11.31 s.
Not explained; watched in abwT113. The agent replay (`agentcmp.py k0 k2`, both or neither): follow-ups 42.56 -> **39.90
s**, side prompts 16.69 -> 16.06 s, every follow-up the same or faster (3.8K tokens at 34K deep 4.40 -> 3.84 s), decode
120.2 / 123.1 tok/s. Card 0's idle rose (658 -> 782 ms in the profiled batch): the expert cards bound more of it now.

## Measured (2026-10-26 03:16 - 04:45: an out-of-memory abort after a long conversation - the stream buffer fits the card now)

- **03:16 the service's engine aborted** with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (the server restarted it). A
  49K-token conversation had grown the TP core's KV rows by ~0.55 GiB on each expert card, and the promotion cache was
  full; the next prompt (4.7K tokens, shallower than `STRATA_XE_PREFETCH_DEPTH`: no prefetch) regrew two expert cards'
  stream buffers to 864 MiB with ~0.9 GiB free. The xe driver evicted to host memory past TTM's limit ("VM worker
  error: -12" in dmesg from 03:08) and a submission failed.
- **`STRATA_XE_STREAM_FIT=1`** (`patches/patch-streamfit.py`, commit c811ed77): a prompt call's stream buffer grows
  only into what its card has free beyond `STRATA_XE_STREAM_FIT_MARGIN_GIB` (0.5) and 100 KiB a token for the call's
  other buffers; a smaller buffer takes more rounds. The replay (a 50.7K needle, then a fresh 4.9K one): both found,
  no fault, the cap engaging (64 MiB) at 0.40-0.80 GiB free. **In the service since 04:30.**
- tsvc2 (`logs/tsvc2-streamfit.log`): needles 8K 1530.9 / 1744.5, 16K 2283.3, 31K 2588.9 tok/s (2581.7 before), all
  FOUND; svcbench 94.6 / 91.8, essay 69.7 / 69.1 tok/s; tool call; hidden 33 GiB at most. The first-pass decode is
  the lowest of the last 14 checks (95.4-102.7 / 73.8-81.3 before) - this engine had just replayed the 50.7K
  conversation (KV replicas grown, experts evicted from the TP cards); the fix touches only prompt calls.
- **The engine binary since 02:55 carries `GGML_SYCL_LI_S16`** (`patch-lis16.py`: the prompt indexer's score matrix in
  fp16, default off), which nobody approved: a step stopped at 02:54 had already applied it and started a detached
  build. Off, its path is the old one (three pointer declarations moved above the new branch). The source is restored
  (04:43, from `logs/tier-rebuild/pre-lis16-lightning-indexer.cpp`); the next ggml rebuild drops it, and the
  re-exported `ggml-sycl-dsv41.patch` (04:44, commit 147963a8: + the AVX2 Q2_0 dot and the op profiler) does not
  carry it.
- **Published** as [nacolias/Larry-DS](https://github.com/nacolias/Larry-DS), branch `larry-xe`, with `larry/` (the
  deploy kit, checks, bench sources, patch scripts, this log). Commit authorship moved to the noreply address, so every
  hash changed (old -> new: `logs/commit-map-noreply-20261026.txt`; the citations in this log are the new ones). A
  clean clone from GitHub, bootstrapped: its repo tree and its ggml tree are identical to the live ones.

## Measured (2026-10-26 00:43 - 01:30: abwT102 - the shared expert on the expert cards, the keep-cache's learned size, the async regrow)

**In the service since 01:21** (commit 9bf548eb; `patches/svc-t102.py`, backup `ds41-xe-engine.sh.pre-t102-20261026`):
`STRATA_SHEXP_TIER=1` at `STRATA_SHEXP_TIER_MIN=4096`, `STRATA_SERVE_KEEP_LEARN_HIST=1`, `STRATA_SERVE_REGROW_ASYNC=1`.
tsvc2 (`logs/tsvc2-t102.log`): every needle found (8K 1422 / 1764, **16K 2139, 31K 2582** tok/s - 00:04: 1604 / 1767 /
2039 / 2446; the first 8K after a start has ranged 1083-1604), svcbench 99.9 / 94.0, essay 78.0 / 70.6 tok/s, the tool
call right, hidden RAM 33 GiB peak, 0 CAT errors. The new build first ran p2k one-shot (`bench/speed/smoke102.sh`:
PPL 2.0032 against the old builds' 2.0035-2.0114; on a failure it would have put the 23:46 binaries back).

**Where the 16K+ prompts are bound** (abwT99 / abwT100): card 0's compute engine is ~100 % busy through layers 0-19
(xe PMU, engmon.sh) with 512- or 1024-column spans, the expert cards ~60 %; card 0's queue (STRATA_GPU_PTL, a warm 16K
batch): 6.6 s, idle 5 %, dense spans 88 % (760 x 7.62 ms) + finish 0.41 ms a span; the expert cards' calls of layers
0-19 71-83 ms (X read from card 0 ~18 ms of it). So the expert cards have ~40 % to spare and card 0 none.

**What splitting card 0 can buy**: a TP split of the attention (heads over the four cards, as decode's TP core does)
has every card repeat the per-query key gather (~0.7 ms a span) and the small replicated ops (norms, hc mixes, q_a /
kv, the indexer); with that the expert cards' load reaches card 0's and the gain stops near -0.6 s at 16K. The FFN
side is cheaper: the **shared expert** (gate | up 5120 -> 2 x 2048, SwiGLU, down: ~0.6 ms of a span) reads the routed
experts' input - already on the expert cards - and its output adds to the routed sum the combine already forms.

**`STRATA_SHEXP_TIER=1`** (`patches/patch-shexptier.py`, `patch-shexpdense.py`, `patch-shexpgen.py`): the engine hands
the TP core's shards to the tier at load (rank k's n_ff rows of gate | up and the matching K blocks of down, planar
Q6_K; rank 0's slice copied to card 1 through the host, 6.45 MB a layer; the slices must add up to the whole n_ff of
every layer, else it stays off); a pipelined part of >= `STRATA_SHEXP_TIER_MIN` columns has its spans write zeros into
the shared expert's columns (`Ds41Dense::set_shexp_skip`) and its tier call compute it: per card and layer the slices
dequantised to fp16 once (fastmv's `dequant_q6k_planar_f16`), per call X -> f16, gate | up GEMM (oneMKL), the clamped
SwiGLU (`xe_swiglu_f16`), down GEMM added into the card's part before the combine.

| `bench/speed/tlppl-shexp.sh` (one-shot, prompt path) | PPL | KL against run a | top-1 | prompt |
| --- | ---: | ---: | ---: | ---: |
| p2k a / b (without) | 2.0009 / 2.0044 | - / 2.17e-3 | - / 98.88 % | 4.83 / 4.52 s |
| p2k s (with) | 2.0067 | 2.43e-3 | 99.02 % | **5.68 s** (tier waits 2.9 -> 4.0 s, dense 1.1 -> 0.9) |
| p8k a / b (without) | 1.6768 / 1.6800 | - / 2.86e-3 | - / 98.91 % | 10.68 / 10.82 s |
| p8k s (with) | 1.6791 | 2.69e-3 | 98.79 % | **10.27 s** (dense 6.3 -> 5.5, tier 1.1 -> 1.5) |

| abwT102 s0 / s1, fresh, second run | without | with (MIN 256) | dense | tier waits |
| --- | ---: | ---: | --- | --- |
| 8K | 4.69 s | 4.66 s | 2.76 -> 2.53 | 1.39 -> 1.58 |
| 16K | 7.26 s | **7.00 s** | 5.66 -> 5.21 | 0.92 -> 1.10 |
| 31K | 12.72 s | **12.09 s** | 10.53 -> 9.49 | 1.19 -> 1.59 |

Card 0 sheds ~0.6 ms a span, but the expert cards' calls grow more than the GEMMs' arithmetic says (p2k: ~14 ms a
call): card 1 holds two slices (its own and rank 0's), and the oneMKL calls are enqueued under the tier's one submit
lock (if a call blocks the host, every card waits). Hence MIN 4096 in the service (16K / 31K prompts' phase-A parts
only). Next: `bench/dqb/mkl_gemm_probe.cpp` (does oneMKL's gemm block, how fast is it), rank 0's slice spread over the
three cards.

**A chat's follow-up turns** (the user's session 2026-10-26 00:04, `bench/speed/turns.py logs/engine.log`): a
conversation growing 25K -> 36K; 131 / 565 / 811 / 1303 new tokens took 0.90 / 1.85 / 1.82 / 2.71 s, 5.2K 4.34 s.
A 565-token turn's 1.79 s batch: dense 0.20 s, **tier waits 1.36 s** (its 512-column calls stream ~27 arena experts
and run ~25 on the CPU pool, ~34 ms a layer), the cache regrow 0.15 s, DSpark's seed 0.04 s. Every turn also shrank
card 0's cache by the calibration (3.5 GiB + 0.45 MiB a column + 1 GiB: 250-330 experts out and back), because the
learned size never applies to a turn that ends deeper than the learned one (keeplearn2's guard), and a chat's next
turn always does.

* `STRATA_SERVE_KEEP_LEARN_HIST=1` (`patches/patch-keephist.py`): the last 8 kept batches - the smallest VRAM a batch
  at least as wide took, times total / max(its end, 16384) for a deeper turn (the attention buffers grow at most
  linearly with the rows), + the 1.5 GiB margin; a batch ending under 1 GiB free forgets them. In the replay: "a
  499-token batch ending at 38080 took 0.61 GiB -> 2.11 GiB for this one (calibrated 4.56)".
* `STRATA_SERVE_REGROW_ASYNC=1` (`patches/patch-regrowasync.py` + `patch-regrowasync-gen.py`): the regrow's copies
  (0.15-0.47 s) left running on the tier's card-0 queue - the TP passes and DSpark use other queues - and published by
  the first TP pass after they finish (`regrow_poll` in `tp_run`: 85-749 experts, 0.15-0.52 s after they were queued).
  Until then a TP pass computes those experts on the CPU pool; `resident()` already counts them, so the route bias is
  what a synchronous regrow sets. No promotion starts while one is pending; every other cache change waits for it.

| abwT102 agent replay (`bench/speed/agentcmp.py`) | k0 / k0b (without) | k1 / k1b (both) |
| --- | ---: | ---: |
| follow-up prompts (16 turns, 56-6,481 new tokens at 7.8-38K) | 45.93 / 45.80 s | **44.98 / 45.03 s** |
| side prompts (fresh 14.9K + 22.2K) | 17.61 / 17.94 s | 17.88 / 17.88 s |
| decode | 116.4 / 120.7 tok/s | 115.6 / 121.2 tok/s |

0.05-0.15 s a follow-up (t08 2.17 -> 2.06, t12 2.77 -> 2.65, t18 1.08 -> 0.99 s); the reply texts differ from run to
run as much without as with (4-5 of 22 reply heads identical either way: the engine is not deterministic).

**Where a follow-up's time goes** (abwT102 t1, `STRATA_XE_CALL_TL`, `bench/speed/calltl2.py`): in a 2-4K-token turn
the layers from 20 on are prefetched (their calls end in 17-34 ms) while layers 0-19 stream on demand (a 2048-column
call 70-75 ms, its second part 50-55 ms): `STRATA_XE_PREFETCH_MIN` is 4096. A 512-column call: layers 0-19 31-43 ms,
20-39 17-27 ms (more arena experts in the lower layers). Next: `STRATA_XE_PREFETCH_MIN=2048` against the replay.

**abwT103 (01:25 - 01:41)**: `bench/dqb/mkl_gemm_probe.cpp` on card 1 - oneMKL's half gemm returns to the host in
0.01-0.02 ms even behind a busy queue (it never holds the submit lock long) and runs the shared expert's shapes at
95-178 TFLOPS (1024 x 1024 x 5120: 0.11 ms; x 5120 x 512: 0.05 ms) - so the p2k one-shot's +1.1 s is not the GEMMs;
most likely the first calls (oneMKL's kernels and the dequant kernel loaded into the tier's context), as the first 8K
needle after a start shows (1422 vs 1764 tok/s the second time). Then `STRATA_XE_PREFETCH_MIN` 4096 -> 2048 on the
agent replay (two runs each): the 3.6-3.8K-token follow-ups 4.57-4.99 -> 3.89-4.34 s, but 2,197 tokens 3.12 / 3.20 ->
3.56 s and 1,711 tokens 2.92 / 2.95 -> 3.01-3.10 s (a batch of B >= the threshold stages every arena expert of every
layer); follow-ups 45.05 / 45.23 -> 42.88 / 42.60 s. **In the service since 01:43: 3072** (`patches/svc-pf3072.py`,
backup `ds41-xe-engine.sh.pre-pf3072-20261026`): the 3.6K+ turns' gain without the 2K turns' loss.

## Measured (2026-10-25 22:15 - 23:20: short prompts' calls - the CPU pool's Q2_0 dots, the submit lock)

* **An AVX2 `ggml_vec_dot_q2_0_q8_0`** (`patches/patch-q20avx2.py`, in the tier's ggml-cpu copy; commit 7e972c82,
  `tools/v41/engine/xe/ggml-cpu-q20avx2.patch`): ggml-cpu had none on x86 (the scalar generic, ~2.5 GB/s a core), and
  Q2_0 is 33 % of the routed gate / up matrices and 44 % of the down ones (`*.experts.json`: gate / up Q3_K 4497,
  Q2_0 5113, ternary 5750; down 4231 / 6682 / 4447). Codes shuffled out and looked up (pshufb), the dot as
  sum(q y) - sum(y) by maddubs / madd, four halves reduced at once, the float steps the generic's in its order:
  `bench/dqb/q20_parity.c` 0 of 12,000 results differ, 5.3 vs 2.46 GB/s a core (`kdot_speed.c`: Q3_K's ggml AVX2 dot
  14 GB/s). `GGML_Q20_GENERIC=1` runs the generic. abwT93 / abwT94: the pool -17 % a call, short prompts even, decode
  +1-2 % - the pool is not a short prompt's bound. Not deployed yet (bit-exact; with the next service change).
* **What the short calls wait on**: a flat perf profile (`bench/speed/perfshort.sh`, abwT95): the six pool workers spin
  ~60 % of the time (`ExpertPool::worker`), the card threads and the main thread barely run - latency, not compute.
  `STRATA_XE_LOCK_PROF=1` (`patches/patch-lockprof.py`, sites by object offset, `addr2line -f`): per 400 calls the
  takers wait ~5 s for the tier's submit lock and the holders hold it ~1.9 s - the GEMM batch's enqueue (gather + up
  to 6 dequant-GEMM launches a phase with their table uploads, ~8 takes a call, 0.35 ms each), the uploads, the
  overlap helper's DMAs, the jobs' quantise / tables. ~5 ms a card thread a call of a call's 20-38 ms "enqueue"
  phase; the rest is waits elsewhere. Next for short prompts: far fewer host steps a call (one table upload, one
  GEMM launch for all quant groups, the routing on the GPU).

## Measured (2026-10-25 21:00 - 22:15: after BAR staging - the CPU pool's share, 1024-column spans, the pipeline's split)

* **`STRATA_XE_CPU_LIGHT=3`** (was 4; **in the service since 22:14**, `patches/svc-light3.py`, backup
  `ds41-xe-engine.sh.pre-light3-20261025`): with BAR staging a light expert's stream costs less than the pool's decode
  of it. abwT87 / abwT88 (fresh, second runs): 1K 2.28 -> 2.17 s, 2K 3.05 -> 2.99, 3K 3.51 -> 3.41, 4.5K 4.13 -> 4.02,
  8K 4.74 -> 4.68; 512 / 16K / 31K even. `CPU_LIGHT=2` and 7 pool workers: no better.
* **1024-column spans again** (abwT88, with BAR): card 0's dense part 16K 5.66 -> 4.59 s, 31K 10.44 -> 8.67 s, but the
  tier's share grows nearly as much - 8K 4.74 -> 4.69, 16K 7.22 -> 7.15, 31K 12.70 -> 12.64, 4.5K +0.35 s. Card 0 and
  the expert cards are about balanced in layers 0-19 now: neither alone buys much. Not adopted.
* **The pipelined batch's split** (abwT89, `STRATA_XE_CALL_TL`): a layer's columns split at a multiple of 512, so a
  513-767-token batch runs 512 + 1..255 - a 530-token prompt's 18-token tier call costs 12-16 ms a layer (its one
  CPU-light expert's activations read back behind card 0's next span). `STRATA_PIPE_MIN_PART` (`patches/patch-pipesplit.py`
  + `patch-pipesplit2.py`, engine, default 0) keeps such layers whole: through the non-pipelined path (abwT90) the FFN
  input goes through the host (530 tokens: 1.97 -> 2.15-2.21 s); on the pipelined path in one part (abwT91) the tier
  work drops by 0.4 s but card 0's dense GPU work no longer hides under the small call (1.98 -> 2.10 s). Left at 0.
  `STRATA_SUBMIT_PRIO=0` (the dense half's submit-lock priority off): even.
* Where a short prompt goes now (530 tokens, ~37 ms a layer): the 512-token tier call ~26-32 ms, bound by its CPU-light
  part (29-61 experts on the pool, 27-49 ms - the pool decodes Q3_K / Q2_0 / ternary at ~3 GB/s a core) next to the
  cards' enqueue-bound part (20-38 ms: one submit lock for every enqueue of the three card threads and the dense half).

## Measured (2026-10-25 18:55 - 20:15: BAR staging - the CPU writes streamed experts straight into VRAM)

Every fresh prompt streams ~60 GB of RAM-arena experts onto the cards. The pinned 24 GiB of the arena goes by DMA; the
pageable rest went through the hot rings (a CPU copy into a pinned ring line, then a DMA a chunk), whose stagers waited
half their time for ring DMAs queued behind the pinned chunks in each card's one copy engine - staging stayed at 14.5 -
16.5 GB/s against the uplink's 28. All four B70s have 32 GiB resizable BARs (`lspci`: BAR 2 = 32 GB, the xe region
query: VRAM 31.89 GiB, all CPU-visible), so the CPU can write into VRAM itself. `bench/dqb/bar_probe.cpp` (a buffer
object made through the xe uAPI on card 1, mmap'd WC):

| | |
| --- | --- |
| CPU -> VRAM through the BAR | **21 GB/s from one thread** (1-6 threads, streaming or plain stores: the CCD's write link) |
| ... while pinned memory DMAs onto cards 1-3 | CPU 6.3 + DMA 20.5 = ~27 GB/s (the x16 uplink) |
| PRIME export of it / Level Zero import | 0 ms and **no host RAM** (a VRAM-only object has no system copy to populate) / works |
| the card reads what the CPU wrote | byte-exact |

(`bench/dqb/uptr_probe.cpp`, a host-pointer import of pageable memory, was written but NOT run: the README's DANGER
note below - that import took card 0's VM down once.)

**`STRATA_XE_BAR_STAGE=1`** (`patches/patch-barstage.py`, `patch-barstage2.py`; commit 30f034a0; **in the service
since 20:05**, `patches/svc-barstage.py`, backup `ds41-xe-engine.sh.pre-barstage-20261025`): the expert cards' flex
regions (the prefetch slots; decode's promotion slots) and stream buffers are such buffer objects (VRAM only,
NEEDS_VISIBLE_VRAM, WC), PRIME-exported and imported into Level Zero as device allocations of their card. The
prefetch stager and the calls' stream helpers write pageable experts straight into them (streaming stores, sfence, a
read back through the BAR - the posted writes are out before the expert counts as staged); pinned experts keep their
DMAs; the slot barriers and the calls' dependencies are as they were. Making one costs its CPU page faults (TTM maps
64 KiB a fault: a 256 MiB stream buffer took 31-48 ms when prefaulted) - they now fault in at first touch, new flex
regions a thread a card, and a stream buffer of the planned size is kept across batches. `bench/dqb/bar_leak.cpp`:
20 alloc / free rounds give the VRAM back exactly. `STRATA_XE_HOTRING_CHECK=1`: 40,000 staged experts read back and
compared, 0 differ (abwT80). Fresh prompts (abwT82 / abwT83, the second of two runs; s = seconds):

| | 512 | 2K | 4.5K | 8K | 16K | 31K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (abwT83, twice) | 2.00 / 1.99 | 2.93 / 3.00 | 4.20 / 4.08 | 5.37 / 5.35 | 7.84 / 7.81 | 13.61 / 13.53 |
| BAR, 4 stagers (abwT82) | | | 3.99 | 5.77 | 7.72 | 16.53 (layer 14) |
| BAR, 2 stagers | 1.95 | 2.86 | 3.90 | 5.08 | 7.45 | 13.34 |
| **BAR, 1 stager** (its default) | **1.94** | **2.90** | **3.84** | **4.94** | **7.43** | **13.19** |

Layers 20-39 ("phase B", staging-bound at every size) take 0.9-1.2 s instead of 1.4-2.2; an 8K batch's pipe trace:
layers 0-19 3.28 -> 2.93 s, 20-39 1.65 -> 1.22 s. Four stagers writing at full speed took the uplink from the calls'
own transfers. The 31K's 4.18 s phase in abwT82 was layer 14's tier call waiting 2.0 + 0.7 s - the Engram layer's
rows read from NVMe (a cold page cache), seen with and without BAR. tsvc2 on the service (`logs/tsvc2-bar.log`):
every needle found (8K **1304 / 1648**, 16K 1959, 31K 2349 tok/s), svcbench 95.4 / 98.0, essay 76.9 / 70.2 tok/s, the
tool call right, hidden RAM 33 GiB peak.

**Then the flex regions** (`patches/patch-barstage3.py`, commit 61a553af, in the service since 20:54): an 8K batch
still spent ~0.36 s more than the base outside its layers - `STRATA_BATCH_TRACE`: prefetch_begin took 481-699 ms a
batch instead of 32-124, and `patches/patch-pfbtrace.py` (`STRATA_XE_ALLOC_LOG=1` prints its parts) showed why: the
computed slot size moves by 10-130 MiB from prompt to prompt, so the BAR flex regions were re-made nearly every batch,
and a new region's CPU pages take 260-420 ms a card to fault in (TTM: 64 KiB a fault). A BAR region now stays while
it is at most 256 MiB too small (the slots shrink to what it holds) or 160 MiB too large; a real change (the TP core's
KV rows growing on the expert cards at the first 16K prompt) still re-makes it once. prefetch_begin: 0-1 ms a batch.
abwT86 (fresh prompts, the second of two runs, against `STRATA_XE_BAR_STAGE=0`):

| | 4.5K | 8K | 16K | 31K | 8K after the 31K |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 4.21 s | 5.33 | 8.01 | 13.38 | 5.39 |
| **BAR staging** | **4.09** | **4.66** | **7.22** | **12.85** | **4.65** |

tsvc2 (`logs/tsvc2-bar3.log`): every needle found (8K **1419 / 1674**, 16K **2091**, 31K **2462** tok/s), svcbench
100.5 / 91.1, essay 75.9 / 72.5 tok/s, the tool call right, hidden RAM 33 GiB peak.

## Measured (2026-10-25 17:50 - 18:55: the slow batch heads were dma-buf exports - fixed)

**Where they blocked.** The stall watch's gdb backtraces (`STRATA_STALL_GDB=1`, abwT76: slow heads of 2.19 and 6.87 s,
both at the first 31K prompt of a fresh engine) have the head's thread in the LM head split (`Ds41XeTier::mv_hook`): an
expert card's memcpy of the quantised input from card 0 inside `DRM_IOCTL_PRIME_HANDLE_TO_FD` (running in the kernel the
whole 6.9 s), and a card's kernel launch in `DRM_IOCTL_XE_EXEC` waiting for that card's VM lock. `bench/speed/xetrace.sh`
(ftrace for an A/B window: function_graph durations over 2 ms of the xe exec / bind / create ioctls, the PRIME ioctls and
the dma-buf / TTM / rebind paths below them; every PRIME / bind / create call in a second instance; the A/B log as
markers; abwT77, two fresh engines) shows what an export costs: `xe_gem_prime_export` -> `ttm_bo_setup_export` ->
`ttm_tt_populate` - xe fills a system-memory copy of the **whole** buffer before it exports it (the ~1 GiB of host RAM
per GiB of peer-resident VRAM found on 2026-10-21), at about 1 GB/s. A warm prompt still did one, 35-102 ms, on the
head's critical path; a re-allocated graph buffer of a GiB or more takes seconds. The split's expert cards copied the
quantised input from, and their rows into, ggml's tensors on card 0 - so whenever a graph's buffers were re-allocated,
the next head exported the new ones.

**`STRATA_XE_HEAD_STAGE`** (`patches/patch-headstage.py`, tier, **default 1**, commit 8447f548): card 0 stages xq + xd in
tier-owned memory (one small kernel); each expert card copies its input from that and its rows back into card-0 staging
(one copy each way instead of 2 + ncols), and a scatter kernel on card 0 writes y. These are byte copies of the same
products: `STRATA_XE_HEAD_STAGE_CHECK=1` runs the direct copies too and compares (abwT78: 0 of 129,280 logits differ).
The split serves only the batch heads; decode's LM head is the TP core's. abwT78 (fresh engines, 8K / 16K / 31K twice):

| | exports in warm prompts | 8K | 16K | 31K |
| --- | --- | ---: | ---: | ---: |
| `STRATA_XE_HEAD_STAGE=0` (as before) | 1 a prompt, 55-102 ms | 6.92 / 5.63 s | 7.96 / **9.26** | 14.26 / 13.50 |
| **staged** | **none** | 7.48 / 5.55 | 7.98 / 7.80 | 13.74 / 13.33 |

The first prompts of a fresh engine still export ~280 buffers (the first 8K) and ~70 (the first 16K) - buffers the tier
and the TP core touch across cards for the first time - part of the first prompt's extra 1-2 s. tsvc2 on the service
(`logs/tsvc2-hs.log`): every needle found (8K 1083 / 1497, 16K 1992, 31K 2334 tok/s), svcbench 101.6 / 89.6, essay
77.9 / 70.6 tok/s, the tool call right, hidden RAM 33 GiB peak. The traces also show an 11.3 s run of xe's rebind
worker on card 0 during one engine start (`preempt_rebind_work_func`: 13 system-memory buffers populated at ~0.9 s
each) - at load only, not in prompts. `patches/patch-alloclog.py` (`STRATA_XE_ALLOC_LOG`): the tier's buffer growths
take 0.3-27 ms - not the tier stalls of a fresh engine's first 16K batch.

## Measured (2026-10-25 13:30 - 16:00: card 0's queue in a fresh prompt - the CPU-light combine, the prefetch's barriers, the batch's VRAM)

unitrace keeps only part of card 0's kernels in a pipelined prompt (107 of ~1,300 key gathers in a 16K trace), so a
new diagnostic shows card 0's queue instead: **`STRATA_GPU_PTL=1`** (`patches/patch-ptl.py`, `patch-ptl2.py`) stamps
the dense queue after every span (attention + router, finish), around the tier's card-0 part and the combine, each
expert card's call (X read, prefetch barrier, GEMM / job rounds, push) and each card's prefetch DMAs, plus the
pipeline's host steps. A stamp is a one-lane kernel writing a sequence number into its own 64-byte line of pinned
memory; a monitor thread notes when it sees it. (The first form had the host clear each flag it saw - a CPU write into
the line the card writes the next flags into - and the card's writes got lost: every ring stopped a few hundred
stamps in.) `bench/speed/ptlsum.py` gives per batch what card 0's queue did and how long it sat empty before each step,
`ptlcall.py` each tier call's cards against its combine. With it off nothing is queued; on, prompts time the same.

**A fresh 8K prompt, card 0's queue (5.42 s):** attention spans 3.37 s (7.65 ms each), finish spans 0.18, the
**combine kernels 0.68 s** (8.5 ms each; 0.4-0.9 ms at 12K / 16K), its own experts 0.3, and 0.86 s empty - waiting for
the layers 0-19 calls (0.35) and for the staging in 20-39 (0.51). The combine was slow because of the CPU-light
experts (`STRATA_XE_CPU_LIGHT=4`, calls of <= 4,096 tokens): their weighted sum went to the combine as a full
n x 5,120 array that the kernel read from pinned host memory over PCIe (84 MB at 4,096 tokens), although those experts
touch a few dozen rows; and their X was read back whole from card 0 (another 84 MB a call, on card 0's link beside the
expert cards' X reads).

| switch | what | result |
| --- | --- | --- |
| **`STRATA_XE_LT_COMPACT=1`** (code default) | the call's light rows listed once: card 0 gathers just those X rows and reads them back, the pool sums one compact row a token, and after the combine a scatter kernel adds them (the same additions in the same order) | combines 662 -> 26 ms a prompt; fresh 8K 6.08 -> 5.67 s (abwT63), 5.4-5.6 since; 16K unchanged |
| `STRATA_XE_DENSE_OFFLOAD=4097` | an 8K prompt's first calls (exactly 4,096 tokens) keep card 0's cached experts on card 0 | -0.2 s at 8K before LT_COMPACT, neutral after - not adopted |
| `STRATA_XE_PF_NOBLOCK=1` (code default 0) | a layer's last call no longer waits for the next layer's staging before its combine (`patches/patch-pfnb.py`: the stager that queues a slot's last copy records its barriers, so they cover exactly its copies) | the wait moves to the next call: 8K 5.52 vs 5.47-5.57 s (abwT65) |
| `STRATA_XE_PINNED_GIB=30` | 6 GiB more of layers 20-39's arena experts pinned (DMA from where they are) | layers 20-39 -0.1 s (staging waits 1.04-1.10 -> 0.92-1.01 s), hidden RAM 40 GiB - not adopted yet |

Numerics (`bench/speed/tlppl-ltpf.sh`, one-shot, CPU-light on): p2k (light experts in every call) KL 2.6e-3 against a
run-to-run floor of 8.7e-3, p8k (the prefetch, PF_NOBLOCK) 4.1e-3 / 3.9e-3, PPL inside the spread.

**In the service since 15:29** (`STRATA_XE_LT_COMPACT` is the code default; tsvc2 `logs/tsvc2-ltc.log`: every needle
found - 8K **1307 / 1470** tok/s (1047 / 1345 at noon), 16K 1949, 31K 2335 - svcbench 101.5 / 97.0, essay 78.6 / 72.6,
the tool call right, hidden RAM 34 GiB peak).

**Where a fresh prompt stands now.** 8K ~5.4-5.5 s: layers 0-19 3.3 s (card 0 ~128 ms a layer; the rest is the
staging: each layer's prefetch lands only ~20 ms before its first call), layers 20-39 1.6 s (staging-bound), 0.55 s
outside the layers (the cache regrow 0.29 s of it). 16K ~7.9 s: layers 0-19 5.3 s, card 0's spans 82 % of its queue
(~7.0 ms a span: fp16 GEMMs ~2.6, attention ~2.2, glue ~2.4), 20-39 1.6 s, regrow 0.47 s.

**A sporadic stall found:** in ~1 of 10-30 prompts the batch's single-row logits took 0.8-7.7 s with every card idle
(PTL: nothing ran). `patch-logitsdiag.py` splits it: the host sat 1.27-1.5 s inside the head graph's *submission*
(the GPU then took 12-25 ms). `patch-stallwatch.py` (on by default, cheap) samples that thread's kernel stack when a
head runs past 300 ms - the next occurrence will say where it blocks.

**The batch's VRAM on card 0.** The keep-cache shrink frees what the calibrated need says (`STRATA_SERVE_BATCH_FIXED_GIB`
3.5 + 0.45 MiB a column + 1 GiB) - right for a start's first big batch, whose pools grow, but later batches find them
allocated: `STRATA_VRAM_TRACE` - an 8K batch took 3.3 GiB of the 8.04 freed for it, 12K 4.5 of 10.0, 16K 5.6 of 11.9.
Every fresh prompt evicted ~4.7-6 GiB of card 0's experts for nothing, streamed them in its batch and uploaded them
again after it (the regrow, before the first token). `STRATA_SERVE_KEEP_LEARN=1` (`patch-keeplearn.py`,
`patch-keeplearn2.py`; default off, in test) sizes the shrink from the last kept batch's low point. Its first form
read the free VRAM after the batch, when ~3.3 GiB of the batch's buffers were already back: the next batch got too
little and an A/B engine ran out of device memory (abwT66, a test config; a clean exit, no GPU fault). The low point
is now sampled during the batch, and the learned size is used only for turns no deeper than the learned one (or 16K),
with 0.6 MiB a column more than the learned batch and 1.5 GiB on top (abwT67: batches kept 1.7-3.5 GiB free at their
low point; regrow 8K 517 -> 266 experts / 0.29 -> 0.16 s, 16K 839 -> 582 / 0.47 -> 0.33 s; fresh 8K / 16K -0.1 s).

**Short fresh prompts (512 - 4.5K tokens)** have no prefetch: a call's arena experts stream on demand, the light ones
(<= 4 tokens) on the CPU pool. Their time is the tier's (a 512-token prompt: dense 0.18 s, experts 1.64 s), and the
pool ends last in most calls (CALL_TL: 41-70 light experts a 512-token call, 30-52 ms, the cards done at 25-40) -
the pool and the streams share DDR4 (~1.2 GB of DRAM traffic a layer). LT_COMPACT already helps them (2K / 3K / 4.5K
-0.2 / -0.33 / -0.4 s). Then:

| switch | what | fresh 512 / 1K / 2K / 3K tokens, s (abwT69) |
| --- | --- | ---: |
| (the service then) | | 2.19 / 2.64-2.72 / 3.66-3.87 / 3.89-3.97 |
| **`STRATA_XE_CPU_LIGHT_PINNED=0`** | the pinned arena's light experts go to the cards by DMA (one DRAM read, no CPU); the pool takes only pageable ones | **2.08 / 2.35-2.51 / 3.24-3.42 / 3.79** (4.5K-16K the same or a little faster, abwT70) |
| `STRATA_XE_LT_QUANT_THREADS=4` (code default 0) | the light experts' activations quantised on helper threads before the job setup | no change |
| `STRATA_XE_CPU_LIGHT=6` (with PINNED=0) | more experts on the pool | 2.14 / 2.53-2.72 / 3.71-3.99 / 3.95-4.08 (worse) |

**In the service since 16:43** (commit 2d1b927a; `patches/svc-keeplearn.py`, backup
`ds41-xe-engine.sh.pre-keeplearn-20261025`): `STRATA_SERVE_KEEP_LEARN=1`, `STRATA_XE_CPU_LIGHT_PINNED=0`. tsvc2
(`logs/tsvc2-kl.log`): every needle found (8K 1301 / 1498, 16K 1969, 31K 2333 tok/s), svcbench 98.3 / 88.5, essay
77.5 / 68.5, the tool call right, hidden RAM 33 GiB peak.

**The slow heads, so far:** the stall watch caught two kinds. (1) The submitting thread blocked ~1 s in the xe driver
at `xe_exec_ioctl+0x547` - disassembled, `down_read_interruptible(&vm->lock)` right after `xe_vm_userptr_check_repin`:
another thread or worker held a card's VM lock for write (a VM bind, or a userptr repin - larry migrates ~11,000
pages a second on average: `pgmigrate_success` 5.8 billion in 6 days, proactive compaction on). (2) The thread ran ~6 s
in user space (no kernel stack, no other thread in the driver) - a busy wait somewhere in the runtime or driver.
`patch-stallwatch2.py` samples the other threads too, `patch-stallwatch3.py` (`STRATA_STALL_PERF=1`) takes a perf
call graph of the stalled thread; `bench/speed/stallhunt.sh` tests whether forced compaction triggers them.
Measured since: forced compaction does NOT trigger them (abwT71: 14 one-shot compactions, ~7M pages migrated during
ten fresh prompts - no slow head; the control config had one). The perf sample of a short one (368 ms) has the thread
spinning in `urEventWait` -> `__sched_yield`: a host-side wait on a GPU event inside the head graph's submission (its
caller is past DWARF's reach; `patch-stallwatch4.py`, `STRATA_STALL_GDB=1`, takes every thread's gdb backtrace).

**1024-column dense spans** (`--prefill-chunk 1024` + `GGML_SYCL_DSV4_SDPA_N=1024`; tabx.sh's new `AB_ARGS`):
card 0's spans get much cheaper - its dense phase 16K 5.6 -> 4.5 s, 31K 10.6-10.9 -> 8.6-9.0 s - but the tier calls
then bound the layers: 16K 7.98-8.10 -> 7.76-7.97 s, 31K 13.7-14.6 -> 13.9 s (abwT73); 4.5K +0.3 s, 512 / 2K the same
(abwT74); one-shot p2k / p8k slower (5.73 vs 4.87-5.11 s, 11.52 vs 10.74-11.06 s) and p8k's KL 3.4e-3 against a floor
of 2.3e-3 (PPL 1.683 vs 1.676 / 1.677). Not adopted alone; with the fp16 FFN input for big calls under test.

The CPU-light part of a short prompt's call, split (`patch-ltprof.py`, `STRATA_XE_LT_PROF=1`, 400 calls of 512-2K
tokens): setup 0.08 s, **X read back 2.6 s** (the compact gather kernel waits for a slot on card 0's busy GPU), job
setup 0.08, the pool 5.85 (18.6 GB/s), sum 0.05 - but the cards end ~2.5 ms after the light part on average now, so
the read back is not on the critical path.

## Measured (2026-10-25 11:20 - 12:05: the prompt attention through oneDNN's fused SDPA)

Card 0 bounds layers 0-19 of every prompt from 16K up (its compute engine 97-100 % busy there), and the attention is
~40 % of its span. oneDNN's graph SDPA (the fused micro-kernel that ggml-sycl's `fattn-onednn.cpp` reaches for GQA)
takes DeepSeek-V4.1's shape when it is read as GQA: batch = the span's 512 queries, one KV head, a group of 64 = the
heads, a query length of 1. The head's **sink** goes in exactly as one more key, [0 .. 0, 1 at dim 512, 0 ..], scored
by an extra Q dimension that holds the sink; as a value row it only touches dimension 512, which is dropped.
`bench/fa/test_sdpa_dsv4.cpp`: **1.04 ms vs 2.30** for the oneMKL path at 640 keys (keys 576 wide; 528 wide: 1.31),
0.47 vs 0.95 at 128 keys, relative error 3.5e-4 (f16 output).

`STRATA_ATTN_SDPA=1` (`patches/patch-sdpa.py`: a ggml op `GGML_OP_DSV41_ATTN_SDPA` + `ggml_dsv41_kgather_pad`, keys
gathered 576 wide with the sink row; `patches/patch-sdpa-dense.py`): the window-only and top-k layers (128 / 640 keys,
layers 0-19) take it for every full 512-query span; partial spans run the oneMKL path over the same buffers, layers
20-39 (all of a context's compressed rows: the count moves) keep KGATHER + oneMKL. One compiled partition a shape,
cached (~0.3 s once after a start). Numerics (`bench/speed/tlppl-sdpa.sh`): KL p512 2.8e-3 / p2k 6.3e-3 against
run-to-run floors of 1.9e-3 / 5.9e-3, PPL inside the floor's spread, top-1 98.6 %.

Fresh prompts (abwT54, the same window): 8K 6.16 -> 6.14 s (its layers 0-19 are bound by the tier calls), 16K 8.32 /
8.23 -> **8.10 / 8.06**, 31K 14.18 / 14.09 -> **13.76 / 13.46** (card 0's dense part 16K -0.35 s, 31K -0.7 s; the
tier calls are the next bound). **In the service since 12:04** (commit c03b1859; `patches/svc-sdpa.py`, backup
`ds41-xe-engine.sh.pre-sdpa-20261025`). tpg.sh and tsvc2 pass: every needle found (8K 1047 / 1345, 16K 1948, 31K
**2330** tok/s), svcbench 96.5 / 92.9, essay 80.6 / 71.5 tok/s, the tool call right, hidden RAM 34 GiB peak.

## Measured (2026-10-25 08:20 - 11:00: why the prompt's expert staging stays near 14 GB/s)

Every fresh prompt streams most of the RAM arena onto the cards (~65-70 GB, 8K or 31K alike): layers 20-39 ("phase B",
the last ~2.9K tokens) are bound by it - each layer's last tier call waits until the next layer is staged, ~84 ms a
layer, **~1.7 s for every prompt size** - and so are short prompts end to end (a 3K prompt is ~4.4 s). Measured:

* `STRATA_XE_HOT_PROF=1` (hot_put's parts): the four stagers spend 52 % of their time waiting for ring slots' DMAs,
  39 % copying (6.5 GB/s a thread), 7 % on the submit lock - the DMA side is the limit, not the copy.
* `bench/dqb/dma_load` (service stopped): pinned host memory onto cards 1-3 at 28.4 GB/s alone; 19-21 GB/s while
  2-6 CPU threads copy memory (DRAM saturates near 33 GB/s: DDR4-3000, two channels); **2.7 GB/s** while the cards read
  168 MB from card 0 by queue memcpy (the tier calls' FFN input). `bench/dqb/ze_engines`: each B70 gives user space
  ONE copy engine (Level Zero ordinal 1; xe keeps bcs8 for itself) and it runs copies FIFO across the card's queues.
* `bench/dqb/p2p_kread`: the same card-to-card reads by a kernel (peer loads, peer access on) run at the same 28.5
  GB/s and leave the DMA 14.6 GB/s; the bytes match.
* `bench/speed/dram-mon.sh` (perf's AMD data-fabric counters; `modprobe amd_uncore`): 20-35 GB/s of DRAM traffic in a
  prompt, peaks ~40. A ring-staged byte costs ~3 DRAM bytes (the CPU's read, the ring line's write-back, the DMA's
  read), a pinned one 1 - staging ~ 33 / (1 + 2 x ring share), 12.4 GB/s at today's 83 %, ~14.5 measured.
* `bench/speed/engmon.sh` (the xe PMU): card 0's compute engine 97-100 % busy through a 16K prompt's layers 0-19, the
  expert cards' ~55 % - at 16K+ card 0's dense span is the limit, at 8K the tier calls.
* `bench/dqb/bench_dqgemm`: the expert GEMMs (xe_dqgemm) run 42-47 TFLOPS alone (96 experts x 200 tokens: ~30 ms); in a
  call the same work took ~65-88 ms - small uploads waiting in the copy engine, host waits between job batches.

| switch | what | result |
| --- | --- | --- |
| **`STRATA_XE_PIN_STREAM=1`** | the pinned 24 GiB re-targeted: card 0's cache below layer 20 (phase A streams it, the regrow refills it) + the next-ranked experts of layers 20.. (phase B's); the cache's layers 20.. (never streamed) pageable | phase B 14.4 -> 16.5 GB/s, 1.70 -> 1.55 s |
| **`STRATA_XE_ZCOPY=1`** | a call's small uploads (row lists, slot maps, the GEMM job / tile tables) by kernels from pinned per-call copies, no host waits on the expert cards | calls ~7 % shorter |
| `STRATA_XE_HOTCOPY_NTA`, `_HOT_PINWAIT` | the ring copies' source prefetched non-temporally; a pinned expert's chunks through the ring's slots | no change |
| `STRATA_XE_QPRIO=1 / 2` | the expert cards' prefetch queue at low priority (2: their call queue high) | phase A -0.15 s at 16K, phase B +0.3 s |
| `STRATA_XE_FN16=2` (+ `_FN16_MIN`) | the FFN input as fp16, read by the expert cards with peer loads (off the copy engine) | phase A's calls get X 3-8 ms sooner, phase B ~4 ms a call slower |
| `STRATA_XE_GEMM_RB`, `GGML_SYCL_LARRY_CPYKERNEL`, `STRATA_ENG_START_ASYNC`, `STRATA_ENGRAM_DIRECT` | bigger GEMM batches; a single graph copy as a kernel; the Engram start on a thread; O_DIRECT Engram reads | within noise |

**In the service since 10:56** (commit 2f95598f; `patches/svc-pinzc.py`, backup `ds41-xe-engine.sh.pre-pinzc-20261025`):
`STRATA_XE_PIN_STREAM=1` and `STRATA_XE_ZCOPY=1`. Against the same window's base (abwT50): fresh 8K 6.39 -> 6.17 s,
16K 8.63 / 8.56 -> 8.31 / 8.25 s, 31K 14.95 / 14.48 -> 14.24 / 14.27 s. tpg.sh and tsvc2 pass: every needle found (8K
1199 / 1330, 16K 1921, 31K 2070 tok/s), svcbench 100.8 (92.3 / 90.4 after the long prompts), the tool call right,
hidden RAM 34 GiB peak.

## Measured (2026-10-25 05:30 - 07:00: the dense span's glue kernels)

With the tier off card 0's critical path, layers 0-19 are bound by card 0's dense parts. Where a 512-column span's
~7.4 ms goes (unitrace, `bench/speed/utprof.py` over logs/ut-utp1): the attention ~3.1 ms (the key gather 0.68, then
ggml-sycl's DSV4 MQA path - Q to f16 0.375, a batched QK GEMM 0.85, the softmax 0.38, a batched PV GEMM 0.82: ~1.1
GB moved at ~490 GB/s), the dense GEMMs ~2.0, hc_post 0.47, the HC mixes' norm + GEMM 2 x 0.3, RoPE 2 x 0.24, casts /
copies ~0.6.

**A fused attention kernel first** (`ggml-sycl/fattn-dsv4-pf.hpp`, `bench/fa/test_mqa_pf.cpp`: a work-group a query,
all 64 heads, 32-key chunks through local memory, an online softmax, O in registers): exact (max |diff| ~1e-3 against
a reference with the same roundings, like the oneMKL path) but **1.7x slower** (3.96 vs 2.29 ms at 640 keys) - its
XMX tiles re-read local memory 4-8 times and its phases run on 16 / 4 / 32 sub-groups between barriers. The oneMKL path
is near the memory roofline already; a fused kernel could save ~1.3 ms a span at most. Not used.

**The glue kernels instead** (each behind a switch, on in the service):

| switch | what | card 0's dense part at 16K |
| --- | --- | ---: |
| (before) | | 235 ms a layer |
| `GGML_SYCL_HC_POST_V2=1` | hc_post one work-item per (dim, token) for all four outputs - it read the residual state once per output (4 x 42 MB a span); the same sums in the same order | 226 |
| `STRATA_ROPE_INPLACE=1` (+ ggml-sycl's rope: an in-place ROPE over part of the row launches only over the rotated pairs) | q's RoPE and the attention output's de-RoPE in place - DeepSeek-V4 rotates the last 64 of 512 dims, the ops copied all 512 | 225 |
| `GGML_SYCL_DSV4_MQA_QV=1`, `_SM=1` | the MQA path's Q conversion four floats a work-item; its softmax a sub-group a row in three passes (a first form held 64 keys a lane in registers and spilled: 272 ms) | ~220 with all four |

Fresh prompts with all four (abwT27, the first big prompt after a start is ~1.5-2 s slower in every config): 16K
10.91 / 8.99 -> 10.62 / 8.58 s, 31K 15.70 / 15.33 -> 15.09 / 14.87 s; against the 05:28 service config (abwT26):
16K 9.39 / 9.50 -> 8.85 / 9.20, 31K 16.46 / 16.19 -> 15.62 / 15.15. Prompt-path numerics (`bench/speed/tlppl-glue.sh`):
KL p512 2.8e-3 / p2k 2.6e-3 against a run-to-run floor of 2.2e-3 / 2.5e-3, top-1 and PPL unchanged.

**In the service since 06:53** (commit 8c67ad95; `patches/svc-glue.py`, backup `ds41-xe-engine.sh.pre-glue-20261025`).
tpg.sh and tsvc2 pass: every needle found (8K 1145 / 1255, 16K 1779, 31K **2090** tok/s), svcbench 101.4 / 92.8,
essay 78.0 / 71.3 tok/s, the tool call right, hidden RAM 33 GiB peak, 0 CAT errors.

**The batch's embeddings** (`STRATA_EMB_PIN=1`, commit df686ca3, in the service since 07:27): `STRATA_DENSE_PROF` now
splits begin_batch - a fresh 16K batch spent 0.46-0.61 s converting its token rows f16 -> f32 on one host thread
(random rows of the 1.3 GB table on the mmap: page faults) and 0.18 s uploading the 335 MB result from pageable memory.
Now the table is copied into pinned host memory at init (1.32 GB, ~0.8 s), a batch's rows are gathered as they are
(f16, 8 threads) into a pinned stage and uploaded async, i_emb is f16 and the init graph casts it (exact): 6-12 ms.
Fresh 8K -0.5 s, 16K -0.3..-0.5 s, 31K -0.9..-1.3 s (abwT31). tsvc2: needles 8K 1202 / 1311, 16K 1877, 31K **2225**
tok/s; svcbench 100.0 / 91.6, essay 79.8 / 74.2, hidden RAM 34 GiB peak. begin_batch's Engram start (hashing and
queueing a batch's row reads) is now its largest part (0.08-0.15 s).

Seen twice in these windows, not explained yet: a one-off stall in a fresh engine's first big prompts (13.6 s outside
the pipeline once - untraced; one 8K tier call of ~4 s once). Larry's /proc/vmstat shows 3.3M compaction stalls and
~110 GB swapped in over 6 days; `STRATA_BATCH_TRACE=1` / `STRATA_XE_CALL_TL=1` will show where if it recurs.

## Measured (2026-10-25 02:45 - 05:15: the dense card out of the prompt calls)

After the night's first round (fresh 16K ~10.2 s), the layers themselves: in layers 0-19 the tier's calls (~140-150 ms
a half) still bounded card 0's dense parts (~120 ms). **`STRATA_XE_CALL_TL=1`** (new, diagnostic: a prompt call's
timeline per card, read at the waits the card threads make anyway, one `call tl:` line a call) showed why: **card 0
was the last card of ~90 % of those calls** - its part ended ~47-50 ms after the expert cards' (16K / 31K; ~25 ms at
8K). Card 0's tier queue runs beside the dense half's own kernels on the same card, and every host round trip there
waited for that card: one streamed 15 MB expert (15 ring chunks, each waited for) took ~120 ms on card 0 against
15-25 ms on cards 1-3, and its GEMM experts + jobs ended 45-110 ms into calls whose expert cards were done at 55-75 ms.
Tools: `call tl` / `pipe trace` summaries per batch and phase (scratchpad `calltl.py`, `pipesum.py`).

**What changed** (all switches off in the code, on in `ds41-xe-engine.sh`; each measured in an A/B window, two runs a
config):

| switch | what | effect (fresh prompts) |
| --- | --- | --- |
| `STRATA_XE_DENSE_LITE=1` | in a pipelined call card 0 reads X where the dense half wrote it (no copy), queues its part without one host wait (pinned per-call staging for the GEMM / job tables; the combine follows on the same queue), and takes no streamed experts (the expert cards take its share; `STRATA_XE_DENSE_LITE_MIN=2048` - streaming on card 0 again in shorter calls - made the 1.5-2.2K turns 0.2-0.5 s slower) | expert waits 16K 2.1 -> 1.3 s, 31K 3.1 -> 1.4 s - but the dense half +0.6 s (card 0's tier kernels now ran beside it): 16K -0.3 s, 31K -0.4 s |
| `STRATA_XE_GATHER_INIT=1` | a card's first gather of a call writes its part (no memset first; same sums) | ~0 to -0.2 s |
| `STRATA_XE_DENSE_INLINE=1` | card 0's part and the combine on the dense half's own queue for the call (in its stream, between its spans) | -0.1..-0.2 s |
| `STRATA_XE_DENSE_OFFLOAD=4096` | in calls of >= 4096 tokens card 0's cached experts are computed by the expert cards from their pinned arena copies (DMA, no CPU copy; kept in the stream buffer for the layer's second call): card 0 runs only the combine | card 0's dense time 7.2 -> 6.7 s at 16K; 16K -0.4..-0.5 s |
| `STRATA_XE_OVERLAP=1` (existing, re-measured) | the expert cards stream their overflow on a second queue while they compute their resident / prefetched experts; its pageable copies now through the card's hot ring (patch-ovlring.py: the bounce path ran ~13 vs ~21.7 GB/s) | 8K -0.4..-0.55, 16K -0.3..-0.5 s |
| `STRATA_XE_PREFETCH_DEPTH` 8192 -> 5000 | batches from empty of 6-8K tokens take the prefetch now (4K ones still not: slower there) | 8K 7.24 / 7.02 -> 6.86 / 6.71 s; 4K unchanged |
| `STRATA_XE_SEED_TAIL_LAYERS=20` | card 0's cache holds the same experts laid out with layers 20-39 first (818 of 1996), so a batch's shrink gives back layer 0-19 experts - which the calls above take from their pinned copies anyway - and keeps the upper layers', whose staging bounds them | 8K -0.3..-0.8 s, 16K / 31K and the agent replay within the noise (2 x 2 replays interleaved) |

Not adopted: `STRATA_XE_FN16=1` (the FFN input to the expert cards as fp16: neutral to worse - at first card 0 was the
bottleneck, later the conversion and widening cost what the copy saved), `STRATA_XE_STAGERS=6` (neutral),
`STRATA_XE_DENSE_OFFLOAD=2048` (= 4096 within the noise), `STRATA_XE_OVERLAP=1` before the dense-card changes
(neutral then).

**Result** (same-window A/Bs, the deployed config before vs now; prompts from empty, two runs):

| | before | now |
| --- | ---: | ---: |
| 4K | 5.97 / 5.16 s | 6.27 / 5.20 s |
| 6K | 6.47 / 6.13 s | 6.24 / 6.04 s |
| 8K | 7.24 / 7.02 s | 6.86 / 6.71 s |
| 16K | 10.35 / 10.37 s | ~9.5-9.7 s |
| 31K | 18.08 / 17.46 s | ~16.2-16.5 s |
| agent replay: all prompts (4 windows) | 85.5-86.9 s | 84.1-87.0 s |
| - its side requests from empty, 15K / 22K | 9.17-9.59 / 13.35-14.38 s | 8.87-9.04 / 13.26-13.43 s |

Decode is unchanged (essay / code / math ms a pass within the noise). The agent replay's prompts are mostly turns at
depth (1.5-4K new tokens on a parked conversation): they gain little - its fresh side requests do. **Where it stands now** (16K): layers 0-19 are
bound by card 0's dense parts (~125 ms a half of 8K columns, ~7.8 ms a 512-column span + ~1.4 ms of finish), the expert
cards end their calls at ~90-100 ms; layers 20-39 by the prefetch's staging (~1.4 GB a layer at ~16 GB/s); outside the
pipeline ~0.75 s a 6K prompt (`STRATA_BATCH_TRACE=1`, new: the batch's columns + `begin_batch` 0.2-0.33 s - host-side
embeddings and maps, the init graph a span at a time -, the prefetch setup 0.01-0.1 s, the logits 0.04 s, the cache
regrow 0.21-0.29 s). Next: card 0's dense work over the cards (the attention block first: its q_b / attention / wo_a /
wo_b are ~3.3 ms of a span, the expert cards have ~30 ms a half to spare), `begin_batch`, the staging rate.

**In the service since 05:28** (commit b880ca88; `ds41-xe-engine.sh` via `patches/svc-denselite.py`, backup
`ds41-xe-engine.sh.pre-denselite-20261025`). tpg.sh and tsvc2 pass: every needle found (8K 1110 / 1254, 16K 1675, 31K
**1929** tok/s - 02:30: 1002 / 1157 / 1545 / 1791), svcbench 93.5 / 96.9, essay 74.6 / 72.4 tok/s, the tool call right,
hidden RAM 32 GiB peak, 0 CAT errors.

Also fixed: `bench/speed/abw.sh` compared dmesg's CAT-error COUNT with the window start's - the ring buffer rotating
old lines out (8 -> 6) read as new errors and left the service down for 9 minutes (03:22-03:31; no request came). It
now counts the kernel journal's CAT errors since the window started (`journalctl -k --since`).

## Measured (2026-10-24 23:30 - 2026-10-25 02:30: fresh prompts - where the time went, the Engram reads, card 0's cache, the first-call stall, a card-0 fault fixed)

The goal: cut the time of fresh 8K-31K-token prompts (new sessions, an agent's side requests). The plan going in was
to split card 0's dense prompt work over the four cards; measuring first showed that card 0's compute was only part of
it, so the fixed costs went first.

**Where a fresh 16K prompt's ~13.4-14.1 s went** (unitrace of the service's engine - `bench/speed/utdrive.py` +
`utprof.py`; engine profiles - `STRATA_ENG_PROF`, `STRATA_XE_PROF`, `STRATA_DENSE_PROF`; `bench/speed/uttl.py`):
- **The Engram rows: 2.4-3.0 s waiting at layer 1** (8K: ~1.5 s). Layer 1 waited for both tables' ~1.2M random reads
  (each row is a weight read plus a scale read 98 GB away), and layer 1's table is on the Sabrent (`/opt`, ~230K random
  4 KiB reads/s at QD 256) while layer 14's is on the WD SN850X (`/`, 590-850K/s) - `bench/engram/iops.c`.
- **Card 0's expert cache released and re-seeded: 0.92 s** for every batch over `STRATA_SERVE_KEEP_MAX` (8192) - and
  the batch streamed those 19.9 GiB of experts too (77 GB of prefetch for one 16K prompt). The keep-cache shrink sized
  the batch with the CUDA engine's constants (4.3 GiB + 0.95 MiB a column = the whole cache at 16K); a 16K batch
  actually takes ~6.8 GiB of card 0 (`STRATA_VRAM_TRACE`: 21.1 -> 14.3 GiB free).
- **The first tier call: ~0.3-0.5 s** a fresh batch, **2.5-7.8 s** for the first big prompts after a load - the
  dense half waiting for the tier's submit lock while the tier's first card-to-card copy of the FFN input mapped a new
  ~5 GiB state buffer for the peers (serve gives the columns back after every batch) and run_big grew its call
  buffers with SYCL allocations (made resident on every peer). An 8K prompt after a 16K one also regrew the stream
  buffers call by call (33 times, ~1.1 s).
- **The layers** (`STRATA_PIPE_TRACE`, `STRATA_XE_PF_TRACE`): layers 0-19 run all columns at ~310-330 ms a layer -
  card 0's dense parts ~120 ms a half (a 512-column span is ~7 ms of GPU; a layer's first span +1.7 ms of fp16 weight
  conversion, cached for the rest) against the tier's calls ~140-150 ms a half (copy the FFN input 17 ms, ~21
  overflow experts streamed in the call ~30-40 ms, the GEMMs ~25 ms, the push 16 ms; the layer's second call's sum
  waits ~17 ms behind card 0's queued spans). The two halves are balanced: splitting only the dense work would hand the
  bottleneck to the tier. Layers 20-39 (the last ~2.7K columns) are bound by the prefetch staging (~120 ms a layer:
  each layer's slot is staged just 3 ms before its call); in layers 0-19 the prefetch is 60-400 ms ahead.

**What changed** (commits de29e40e, 5b901f34, 1d9e8d3f; the service's settings since 02:27):
- **A paged copy of layer 1's Engram table on the fast drive**: `/srv/dsv41/engram-epg/engram-L1.epg` (104.9 GB; the
  repo's shard is untouched) - 4 KiB pages of 15 rows, each its 256 weight bytes, 8 scale bytes and 8 zero bytes, so
  a row is one aligned read. `bench/engram/engram_epg.py convert|verify` wrote and checked it (all 384,006,168 rows,
  0 bad bytes); the engine uses `engram-L<layer>.epg` from any `--engram` dir before the shards (that dir first in
  larry-ds41.json). `bench/engram/test_engpg` diffs the reader's rows against the shards on every path: identical.
- **Each Engram slot waited for at its own layer** (slot 0 at layer 1, slot 1 at layer 14), the reads queued slot by
  slot. With both: **0 ms** of Engram wait at 8K-31K (was 1.5-3.0 s).
- **Keep-cache for every batch, sized for the xe engine** (`STRATA_SERVE_KEEP_MAX` 8192 -> 32768): the shrink uses
  `STRATA_SERVE_BATCH_FIXED_GIB` / `_COL_MIB` (3.5 GiB + 0.45 MiB a column, ~5 GiB over what a 16K batch takes)
  instead of the CUDA engine's constants: a 16K batch keeps ~9 GiB of the cache and regrows the rest in ~0.45 s.
- **The expert cards' prompt-chunk parts to card 0 in fp16** (`STRATA_XE_COMBINE` 1 -> 2, an existing mode: half the
  bytes over card 0's link; fp16 has more mantissa than the model's bf16): 16K 10.54 -> 10.17 s, 31K 17.69 -> 17.41 s.
- **The FFN input (o_fn) in its own buffer for the engine's life** (335 MB of card 0), and the expert cards' call X
  buffers, GEMM scratch and stream buffers in card-local memory (no peer mapping), the stream buffer grown by half its
  size at least: the dense half's submit-lock waits 236-7767 -> 4-543 ms a prompt.

| fresh prompt (from empty, 16 tokens out) | before | after |
| --- | ---: | ---: |
| 8K (first / second after a load) | 9.5-12.2 s | 8.1 / 7.2 s |
| 16K | 13.4-14.1 s | **10.1-10.2 s** |
| 31K | 19.9-23.1 s | **17.2-17.6 s** |
| agent replay (all prompts; decode; keep-cache to 32K, f32 parts) | 96.5 s; 120.6 tok/s | **86.5 s**; 120.2 tok/s |
| - side requests from empty, 15K / 22K | 11.1 / 16.4 s | 9.4 / 14.2 s |

tsvc2 at the service's settings (02:30): every needle found (8K 1002 / 1157, 16K **1545**, 31K **1791** tok/s; the
evening before: 847 / 1041, 1363, 1694), svcbench 95.9 / 92.5, essay 74.6 / 72.6, the tool call right, hidden RAM 32 GiB
peak, no new CAT errors.

**Note on exactness checks**: two runs of the same engine on the same prompts give last-row logits that differ by up to
~1-3 (argmax the same): the engine is not bit-reproducible run to run, so changes are checked component by component
(the Engram rows above) and with needles, not by diffing logits across runs.

**Card-0 GPU faults, found and fixed (2026-10-25 01:24 / 01:54; patch-pcrace.py)**: card 0 "Engine memory CAT error
[18]: class=ccs" + DEVICE_LOST in the first card-0 tier call of a 16K batch right after a keep-cache shrink (test
engines on port 18081; no user request). Reproduced (`STRATA_XE_TP_EVICT_KEEP_GIB=1.5` + `STRATA_SERVE_KEEP_MAX=32768`,
prompts 8K, 8K, 16K) with every allocation logged (`STRATA_ADDR_LOG=1`, patch-addrlog.py; `GGML_SYCL_MEMTRACE=2`): the
faulted addresses (dmesg prints them without the sign extension: 0x0000d55b5ce98000 is device VA 0xffffd55b5ce98000)
lay in a 1 GiB card-0 cache slab the shrink had just freed - it freed 11 slabs but released only 294 experts. The
chain: the TP core's KV-room eviction made hot expert-card experts arena experts; decode ran them on the CPU pool and
queued their promotions; an 8K prompt from empty (no prefetch, so no `pc_drop`) let the promotions live on while card
0's cache regrew - taking some of the same experts; the next decode completed those promotions, which moved the
experts to an expert card's slot without looking; card 0's cache list still held them, and the shrink's walk of that
list's tail stopped at the first moved entry but freed the slab anyway. Fixed four ways: the seed / regrow skips an
expert a promotion holds; a promotion that completes for an expert placed meanwhile leaves it there; the shrink evicts
every cached expert in a slab it frees (a full scan) and drops stale entries; the expert cards' KV-room eviction finds
a slab's experts anywhere in its list. The same chain could hit the service's 8K keep-cache shrinks too (rarer: it needs an
eviction first). With the fix the reproduction ran 12 prompts clean (every shrink now releases ~93 experts a slab it
frees); recovery both times: GT reset x4, copy_probe clean.

Not adopted: `STRATA_XE_RESERVE_GIB=4.0` (bigger prefetch slots on the expert cards): within the noise.
`STRATA_XE_PREFETCH_DENSE=1` (card 0 takes the layer's overflow experts; with keep-cache it has no room, without it
16K 10.66 s = keep-cache's 10.54): not switched.
`STRATA_SERVE_BATCH_FIXED_GIB=8.0` for the keep-cache: shrinks more than a batch needs (8K +0.6 s).

## Measured (2026-10-24 ~22:30 - 23:05: two parked conversations; pinning the streamed experts)

**Two park slots** (`STRATA_SERVE_PARK_SLOTS`, 2; commit 54ddd62e, in the service since 23:05). With one slot, a parked
conversation nobody continued (an agent session that ended right after a side request; a test) kept the slot, and the
next side request could not park the live conversation - its next turn read everything again. Now a side request
parks the live conversation into a free slot, else the one parked longest ago; a request unparks the slot it continues
best; a park saves the whole conversation when it fits `STRATA_SERVE_PARK_MAX` (262,144), so no later write can spoil
it. The agent replay right after the prompts test (which left two conversations parked): prompts **115.7 -> 96.5 s**,
both main turns after the side requests resumed (7.5 / 5.9 s instead of 15 / 22 s), decode 120.6 tok/s; tpg.sh: every
needle found, the tool call right.

Not adopted: `STRATA_XE_PIN_SKIP_GIB=18` (the pinned arena moved past what card 0's cache holds, onto the experts
prompts stream): the expert phase -0.1..-0.2 s a turn, but card 0's re-seeds and regrows went through the bounce
(+0.2..+0.7 s): replay prompts 115.7 -> 120.6 s.

## Measured (2026-10-24 ~18:10 - 20:40: a 1M-token context)

The user asked for DeepSeek's full context. The KV cache is ~3 KB a token: the TP cards' replicas take 784 MiB each at
262K and 3,184 MiB at 1,048,576 (mapped as the context grows); card 0's own copy is allocated at load (+2.3 GiB at
1M). `--ctx 1048576` worked as far as 490K tokens out of the box, then needed two fixes (commit 880b88ff):

* **The prompt indexer in column slices** (`STRATA_LI_CHUNK_MIB`, 512): its [cap, n] tensors (the visibility mask
  chain, the F16 mask, the scores, the candidate mask, top-k's workspace) grow with the context - ~9 GiB on card 0 for
  a 512-column span at 1M, and a 984K-token prompt's last batch (the upper layers, phase B of enc-skip) failed with UR
  OUT_OF_RESOURCES (card 0 free 10.47 -> 1.46 GiB). Every query column ranks on its own, so slices give the same ids;
  up to cap 262,144 it is the old graph. tpg.sh with 16-column slices everywhere: every needle found, the tool call right.
* **The expert cards' KV room evictable** (`STRATA_XE_TP_EVICT=1`): the tier lent all of the replicas' future VRAM to
  the flex region at load, so at 1M the cards held 757 fewer experts (6,936 of 7,693) and a fresh agent replay ran
  ~8 % slower. Now only `STRATA_XE_TP_DEFER_FLEX_GIB` (0.8: what 262K needs) is lent; the rest holds experts in each
  card's last slabs, and when a long context needs the room `tp_make_room` evicts a slab at a time (its experts read
  from the model file into a host buffer - arena experts from then on; ~0.35 s a slab). Arena copies made at load took
  ~9 GiB of RAM: larry swapped and the replay's prompts ran 147 s - hence the copies at eviction time.

| `--ctx 1048576`, eviction on | prompt | decode after it (400 tokens) |
| --- | ---: | ---: |
| needle 129K from empty | 80 s (1,610 tok/s) | 90 tok/s |
| needle 245K | 166 s (1,481 tok/s); through the service 156 s | 78 tok/s |
| needle 490K | 393 s (1,247 tok/s) | 81 tok/s |
| needle 594K | 543 s (1,095 tok/s) | 71.5 tok/s |
| needle **984,543** | **1,140 s (863 tok/s)**, found | **61.7 tok/s**; then an 11.7K-token turn on top: 34 s |

The first reply after a long prompt waits for the TP cards to map and copy the KV rows (and evict): 4-7 s at 600K-1M.
The 262K config does 129K in 89 s and 245K in 158 s - long prompts cost the same either way.

**The cost at ordinary lengths, and how it went away.** Card 0's own KV state is allocated for the whole context at
load: 2.3 GiB more at 1M, which its expert cache gave up (1,850 -> 1,669 experts) while the replicas kept their 2.5 GiB.
Agent replay at 1M (eviction keep fixed: 0.25 GiB - the first version kept 1.5 and evicted ~570 experts after every
first prompt): prompts 103.3 s, decode 112.6 tok/s, 12.4 CPU-pool experts a pass, against 99.0 s / 123 tok/s / 8.3 at
262K. Giving card 0's cache the replicas' VRAM (`STRATA_XE_DENSE_RESERVE_GIB` 4 -> 1.75, `STRATA_XE_REPLICA_GIB` 2.5
-> 0.25; the replicas had used ~2.5 of that reserve anyway): **prompts 99.0 s**, decode 115.6 tok/s, 10.2 a pass -
and 2.5 GiB less pinned host memory (tsvc2: hidden 41 GiB, MemAvailable >= 21 GiB).

*Tried and dropped*: card 0's KV state in Level Zero virtual memory mapped as the context grows
(`STRATA_DENSE_LAZY_KV`, commit cc228cde, default off - **unsafe**): mapped after load, card 0 faulted (DEVICE_LOST)
in the 16K batch starting at position 524,288, twice; mapped all at load a 594K needle passed; with the TP pager off
it faulted at the first request (CAT errors, class=ccs). The other cards' copy engines read card 0's virtual memory
exactly, grown pages too (`bench/engine1/vmm_peer_read`), so something else reads past the mapped rows. That last test
left the service down 21:50-22:06 (the A/B window keeps it down on new CAT errors): GT reset of all four, copy_probe
clean, service back, needles found; no request came in meanwhile.

**In the service since 20:35** (`larry-ds41.json --ctx 1048576`, `ds41-xe-engine.sh` `STRATA_XE_TP_EVICT=1` and
`STRATA_SERVE_PARK_MAX=262144`; backups `.pre-ctx1m-20261024`), the cache settings since 22:25 (backup
`.pre-cache1m-20261024`; tsvc2: needles 8K 847 / 1041, 16K 1363, 31K 1694 tok/s, svcbench 91.0 / 93.2, essay 79.7 / 72.9,
hidden 41 GiB, MemAvailable >= 21 GiB; a 245K needle through the service 155 s). Before them: tsvc2: every needle found (8K 866 / 734, 16K 1199, 31K
1656 tok/s), svcbench 101.2 / 86.0, essay 77.5 / 69.2, the tool call right, hidden 44 GiB, MemAvailable >= 13 GiB.
Back to 262K: restore the two backups, `larry-mode restart`.

## Measured (2026-10-24 ~16:35 - 18:10: an agent session replayed; the prompt's expert copies through rings in the L3)

**The agent replay** (`bench/speed/agentx.py`, a tabx.sh / abw.sh test "agent"): the shape of the 14:13 session - a
7.8K-token first prompt (system text and tool list), then 18 turns of tool results of 17-6,500 tokens with the context
growing to ~38K, two side requests of another conversation from empty (15K / 22K) in between, and one long copy-heavy
answer at the end - on this machine's own source tree, never a user's prompts. The assistant turns in the history are
fixed texts, so every configuration reads the same prompts. Per turn: the engine's prompt time and phases, decode tok/s,
DSpark passes, CPU-pool experts a pass. On the service as of 15:22 it took 130 s: **prompts 108 s, decode 21 s** (the
model answers the tool results briefly, ~100 tok/s at 3.3 tokens a pass, 10 CPU-pool experts a pass - not the 22-54 of
the real session, whose wide lookup passes are gone). In every batch turn the expert phase is 50-80 % of the time.

* `STRATA_SERVE_BATCH_MIN=64` (turns of 64-127 new tokens as a batch, not passes): slower - 90 tokens 0.82 -> 0.97 s,
  99 tokens 1.07 -> 1.16 s. It stays 128.
* **Why the expert phase is slow** (`bench/dqb/stage_ring`, 6 GiB of pageable "experts" onto cards 1-3): the tier filled
  a 64 MiB bounce half (on demand) or a 1.5 GiB layer stage (prefetch) completely before any DMA started, so every line
  was out of the caches by then - read from DRAM, written back, read again by the DMA: **13.0 GB/s** end to end, 4
  threads or 6. A 16 MiB pinned ring filled with plain stores, each 1 MiB chunk DMA'd as soon as it is copied:
  **21.7 GB/s** - the card's read is served from the L3 (the same ring with streaming stores: 13.7; a 32 / 64 / 256 MiB
  ring: 16.5 / 11.6 / 9.4). ~21.7 GB/s is the 5800X's CCD -> IO-die write link; pinned memory DMAs at 28.4 (the x16
  uplink).

**`STRATA_XE_HOTRING=1`** (commit 6b47745d; `_KB` 1024, `_N` 4, the service 2): a ring per card thread (on demand) and
per prefetch stager; the stagers put each planned expert of a slot on its card themselves (pinned ones as direct
chunked DMAs) and a barrier on each card's copy queue ends the slot. With it the prefetch's two 1.5 GiB pinned layer
stages hold nothing - `STRATA_XE_HOTRING_NOSTAGE` (default 1) drops them and gives their budget to the arena (23.9 GiB
pinned now). `STRATA_XE_HOTRING_CHECK=1` reads every streamed / prefetched expert back and compares it: 20,000+
compared, 0 differ.

| | before | ring, 4 slots | ring, 2 slots | 2 slots, no stage (**in the service**) |
| --- | ---: | ---: | ---: | ---: |
| agent replay: all prompts | 127.0 s | 120.5 | 118.5 | **116.5** |
| - 3.6K-token turns at 8-34K (t03 / t07 / t15) | 7.31 / 6.97 / 7.31 s | 6.30 / 6.08 / 6.61 | 6.14 / 5.94 / 6.45 | **6.01 / 5.92 / 6.27** |
| - 1.4-2.2K-token turns (t09 / t11 / t13) | 4.63 / 4.29 / 3.93 s | 4.54 / 4.14 / 3.91 | 4.49 / 4.05 / 3.74 | 4.42 / 3.96 / 3.70 |
| - side requests from empty, 15K / 22K | 12.18 / 17.94 s | 11.30 / 17.04 | 11.27 / 16.66 | **11.12 / 16.36** |
| prompt 3000 from empty (2 runs) | 7.79 / 6.01 s | 6.80 / 5.66 | 6.95 / 5.52 | 6.99 / 5.64 |
| prompt 8000 from empty | 11.09 / 9.58 s | 9.87 / 8.80 | 9.56 / 8.75 | 9.87 / 9.25 |
| prompt 16000 from empty | 14.39 / 16.13 s | 13.22 / 12.95 | 13.06 / 12.80 | **13.31 / 12.87** |
| prompt 31000 from empty | 21.78 / 20.90 s | 20.21 / 19.81 | 20.25 / 19.81 | **20.94 / 19.45** |
| the CPU-light pool's rate (it shares DRAM with the copies) | 16.3 GB/s | 18.1 | 18.9 | 19.1 |

The on-demand turns (no prefetch: < 2,560 new tokens, or ending before 8K) gain least: their copies run on the card
threads between their own kernels, and the CPU-light pool takes the light experts anyway. Those four runs had the
replay's main conversation re-read twice from empty after the side requests (a stale park from the prompts test before
it, see below), the same in every column. Without that (the "pk" run, the replay alone): **prompts 99.0 s, decode
26.0 s, 127 s in all**; the closing 1,500-token copy at 38K tokens of context ran **166 tok/s** (5.3 tokens a pass).

Not adopted: the prefetch for every batch of >= 1,024 tokens at depth (1.4-2.2K turns 0.2-0.6 s slower: on demand with
the CPU-light pool wins there). `STRATA_XE_PREFETCH_MIN=2560` (the lower layers prefetched too for 2,560-4,095-token
turns at depth): 3.6K turns -0.2..-0.3 s, all prompts 99.0 -> 98.0 s - within the noise; not switched.

**One parked conversation.** The server keeps one conversation besides the live one (`STRATA_SERVE_PARK`): an agent's
side request parks the main conversation and the next main turn unparks it (0.02-0.06 s for a 12-30K conversation). A
parked conversation nobody continues keeps the slot until a request writes past its saved rows, and while it does,
the next side request cannot park the live one - in the A/B the prompts test left one, and the replay's main
conversation (18K, then 34K tokens) was read again from empty (+7 and +16 s). The user's own 14:13 session parked and
unparked fine three times. Also changed (patch-parkcap.py, in the service since 18:08): a park saves only the
conversation's own rows - it saved every position the side request could write, and an agent's side call asks for 32,000
tokens (a 12,588-token conversation was saved as 47,012 positions).

**In the service** since 17:34 (`ds41-xe-engine.sh`: `STRATA_XE_HOTRING=1`, `STRATA_XE_HOTRING_N=2`; backup
`ds41-xe-engine.sh.pre-hotring-20261024`), the park change since 18:08. tpg.sh x2 and tsvc2: every needle found (8K
858 / 1045, 16K 1366, 31K **1702** tok/s - this afternoon 728 / 995, 1247, 1605), svcbench 95.9 / 94.2, essay 79.7 /
72.3, the tool call right, hidden RAM 42 GiB peak, 0 CAT errors.

## Measured (2026-10-24 ~14:40 - 15:25: lookup drafts on the TP core, shallow prompts without prefetch, keep-cache to 8K)

Where the time went in a real agentic session (the service's engine log, 14:13-14:21: 19 turns of a coding agent, the
context growing from 12K to 43K tokens; `bench/speed/turns.py` prints such a table): prompts 112 s, decode 291 s, and
the decode ran 47-74 ms a DSpark pass (the speed table's 30). Two causes:

* **Lookup drafts took the old path.** `STRATA_SERVE_LOOKUP=14` (upstream's serve default) drafts up to 14 tokens copied
  from the context when its last tokens occurred there before - an agent's edits and quoted tool output. A verify pass
  of more than 7 rows does not fit engine1's TP core, so it ran the old dense path, which pages card 0's copies of the TP
  cards' weights in for every layer: ~0.1-0.2 s a pass, about twice a TP pass's time a token. 10-20 % of the session's
  passes were such passes - roughly a third of its decode time (estimated from the per-request totals).
* **The CPU pool:** 22-54 CPU-pool experts a pass over all passes, the wide lookup passes included (the essay
  benchmark: 7-8).

`STRATA_LOOKUP_TP=1` (commit 605ccc23, default 0) caps a lookup draft so its pass stays on the TP core (7 rows with the
token and a re-fed one). Lookup drafts are verified like DSpark's (each accepted with the model's own probability), so
the output distribution is unchanged. `STRATA_XE_PREFETCH_DEPTH=8192` (same commit): a prompt batch that ends before
position 8192 takes no prefetch (shallow batches are bound by their experts; it also keeps the decode's promotion
cache, which a prefetch drops). `STRATA_SERVE_KEEP_MAX=8192` (an existing knob, was 4096): a batch of up to 8192 new
tokens gives back only the part of card 0's expert cache it needs, instead of closing the cache and re-seeding all of
it after the prompt (~1.05 s).

A/B through the server (`bench/speed/abw.sh`: one window, the service held, each configuration the service's server
and engine on port 18081 with clocks pinned; `abx.py`: greedy, no thinking; prompts the mean of 2):

| | before | + depth 8192 + lookup on TP | + keep-cache 8192 (**in the service**) | lookup off | + flex 2.5 / reserve 5.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prompt 3000 tokens from empty | 6.83 s | 7.24 s | 6.46 s | | 6.66 s |
| prompt 4400 from empty | 9.04 s | 8.13 s | **6.80 s** | | 8.01 s |
| prompt 6000 from empty | 9.59 s | 8.81 s | **7.91 s** | | 9.12 s |
| prompt 8000 from empty | 10.50 s | 10.12 s | **9.34 s** | | 10.61 s |
| a copy-heavy edit (a 10.4K-token file reproduced with a comment per function, 1200 out), 2 runs | 62.2 / 63.9 tok/s | **183.3 / 190.5** | | 165.1 / 175.9 | 183.0 / 191.0 |
| - its passes | 9.92 tok, 157 ms | 5.77 tok, 31 ms | | 4.77 tok, 28 ms | 5.80 tok, 31 ms |
| essay, 400 out | 73.3 tok/s | 77.8 | | | 75.3 |
| code, 400 out | 102.2 | 109.9 | | | 115.8 |
| math solution, 400 out (it quotes its own lines: lookup) | 92.6 | **117.2** | | | 117.0 |

The 3000-token row's first run varies by ~1.5 s in every column (the first prompt after the warm-up): the second runs
are 6.11 / 5.96 / 5.92 / 5.83 s. Without lookup drafts the edit is already 2.7x the old setting (DSpark's own drafts copy
well); the capped lookup adds 8-11 %.

**Not adopted: a bigger promotion cache.** `STRATA_XE_FLEX_GIB=2.5` with `STRATA_XE_RESERVE_GIB=5.0` (690 slots instead of
372; the expert cards' static share 1.5 GiB smaller each): decode within the noise (CPU-pool experts a pass 13.6 -> 16.3
essay, 7.7 -> 10.4 code, 14.0 -> 14.7 math) and 6-8K prompts 0.3-0.5 s slower. A replay of the usage profiles
(`bench/engine1/flexsim.py`, LFU steady state) had promised -1.3 / -0.1 / -5.5 CPU experts a pass for essay / code /
math: the 400-token decodes never reach it, and the experts the bigger cache displaces cost as much as it saves.

**In the service since 15:22** (`ds41-xe-engine.sh`: `STRATA_XE_PREFETCH_DEPTH=8192`, `STRATA_LOOKUP_TP=1`,
`STRATA_SERVE_KEEP_MAX=8192`; backup `ds41-xe-engine.sh.pre-afternoon-20261024`). tsvc2 at 15:23: needles 200 / 8K / 8K / 16K / 31K found (8K 728 / 995, 16K 1247, 31K 1605 tok/s),
svcbench 95.5 / 92.4 tok/s (10:57: 86.4 / 81.9), essay 77.3 / 72.6 (77.6 / 67.2), the tool call right, hidden RAM 41
GiB peak, no CAT errors.

A request that arrived during the window would have got a 503 from `bench/speed/yieldsrv.py` (standing in on :18080)
and brought the service back; none came.

## Measured (2026-10-24 ~08:25 - 10:55: short prompts' light experts on the CPU pool, the prefetch from 4096 tokens)

A prompt of fewer than 2,048 new tokens has no prefetch: after each layer's router, the experts it uses that are on no
card stream to the cards from the RAM arena - ~2,000 of them for 512 tokens. 70 % of the arena is ordinary (pageable)
memory, so such an expert goes through larry's DDR4 three times (the card thread's copy into a pinned bounce reads and
writes it, the DMA reads it again) to serve a token or two: the expert phase (2.0 of a 512-token prompt's 2.7 s) was
bound by host memory while the CPU pool sat idle. `STRATA_XE_CPU_LIGHT=4`: an arena expert the call would stream that
serves at most 4 tokens runs on the pool beside the cards instead (one read of its bytes); its weighted sum joins the
dense card's combine as one more part. Batches of more than 512 tokens run pipelined with their X on card 0 only: the
tier reads it back once a call.

| service settings, DSpark, clocks pinned, mean ± sd of 5 (`bench/speed/tab_all.sh`) | T = 0 (before) | T = 1 | T = 2 | **T = 4** | T = 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp512 @ d0, tok/s | 165.9 ± 9.9 | 190.4 ± 7.6 | 213.6 ± 12.8 | **230.8 ± 11.2** | 215.2 ± 10.8 |
| pp512 @ d8192 | 158.5 ± 7.2 | 178.0 ± 10.6 | 201.4 ± 10.8 | **221.5 ± 6.4** | 202.9 ± 10.6 |
| pp512 @ d32768 | 154.3 ± 6.4 | | | **218.6 ± 5.5** | |
| pp160 @ d0 | 90.8 ± 2.9 | 110.5 ± 3.1 | 127.7 ± 2.0 | **124.5 ± 3.1** | 120.6 ± 1.7 |
| 2,047-token prompts | ~390 | | | **~490** | |
| a 31K prompt from empty | 23.7 s | | | 23.1 s | |

T = 0 again at the end: 165.5 / 157.3 / 90.8 (no drift). At T = 8 the pool's multi-token jobs crowd the cards' host
threads (the cards finish later although they stream 6 experts a layer instead of 13). Turns of 32-127 tokens stay on
the passes (95 tokens as a batch with the pool: 92 tok/s against 98-102).

The prefetch (every arena expert of a layer staged onto the cards ahead of its call) took prompts from 2,048 tokens,
and there the pool hardly runs (2,048 tokens ~345 tok/s either way). From 4,096 (`STRATA_XE_PREFETCH_MIN`) whole
prompts of 2-3K tokens run on demand with the pool, but alone that took the prefetch from a long prompt's upper layers
too (enc-skip: layers 20-39 run only its last ~2,560 tokens, over the whole context; card 0's attention is the
bottleneck there and the pool's activation quantisation slowed its host thread) - a 31K prompt +5 s. So those layers
keep their own threshold (`STRATA_XE_PREFETCH_MIN_B` 2048, commit 8c430ac5):

| T = 4, 5 runs | prefetch from 2048 | from 4096, upper layers from 2048 |
| --- | ---: | ---: |
| 2,047-token prompts, tok/s | 436 ± 79 | **502 ± 20** |
| 2,900-token prompts | 455 ± 10 | **588 ± 35** |
| pp512 @ d32768 | 221 | 221 |
| a 31K prompt from empty | 23.0 s | 23.4 s |

In the service since 08:56 (tsvc2: needles found, the tool call right, svcbench 84.8 / 87.2, essay 79.1 / 76.2,
prompts 8K 763 / 872, 16K 1106 - this needle's card-0 dense phase has varied 6.6-9.8 s across runs, its expert phase
2.78 s as before - 31K 1615 tok/s).

Prompts of 3-4K tokens are cut by enc-skip, so only their upper layers reached a threshold, yet they were prefetched
whole (the range always began at layer 0): now layers 20-39 only, the lower ones on demand with the pool (commit
4a1d29ad, identical to before whenever the two thresholds are equal). At depth 0 no prefetch at all pays even more,
but not for large calls:

| 5 runs (4 for 6K-16K) | prefetch from 2048 everywhere | the service (layers 20-39 for 3-4K) | none (the pool up to 16K-column calls) |
| --- | ---: | ---: | ---: |
| 2,047 tokens, tok/s | 446 ± 85 | 520 ± 15 | 527 ± 15 |
| 3,500 tokens | 510 ± 15 | **559 ± 23** | 642 ± 28 |
| 6,000 tokens | | 650 ± 14 | 741 ± 26 |
| 10,000 tokens | | 980 ± 43 | 1014 ± 42 |
| 16,000 tokens | | 1301 ± 86 | card 1 out of resources |

Without the prefetch a 16K prompt's 8K-column calls stream on demand: card 1 ran out of VRAM
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`), its copy engine timed out and the driver reset it (10:25; that test config only
- the engine was stopped, all four GTs reset, copy_probe clean on each). Not deployed; a depth rule for batches below
~8K tokens is drafted (`patches/patch-pfdepth.py`, `STRATA_XE_PREFETCH_DEPTH`, not built).

**A regression and its fix (10:55).** The speed table at 10:15 had DSpark's long prompts from empty 2-10 s slower
(31K 23.2 -> 25.3 s, 97K 62.5 -> 72.4 s), all of it card 0's dense phase (97K 50.1 -> 60.1 s); plain unchanged. The
pool's weighted sum went into a buffer on card 0 grown to the largest call (~70 MB after a 6.7K prompt), and with
DSpark's draft that took card 0 under ggml-sycl's floor for its fp16 weight cache (3 GB free): the prompt GEMMs then
dequantise on every call. The combine now reads the sum from its pinned host buffer (commit 7ef5fd41):

| DSpark, 3 runs, first messages at the speed table's depths | pool off | as it was | **now** |
| --- | ---: | ---: | ---: |
| 7,170 / 31,758 / 97,294 tokens from empty | 11.8 / 24.3 / 63.7 s | 11.2 / 23.6 / 68.7 s | **11.4 / 22.3 / 62.5 s** |
| the 97K prompt's dense / experts phases | 48.3 / 9.0 s | 56.5 / 6.4 s | 49.9 / 6.4 s |
| pp512 @ d8192 / d32768 / d98304, tok/s | 154 / 151 / 144 | 222 / 212 / 194 | **220 / 216 / 196** |

In the service since 10:57 (tsvc2: needles found, the tool call right, svcbench 86.4 / 81.9, essay 77.6 / 67.2, prompts
8K 765 / 873, 16K 1259, 31K 1616 tok/s, hidden RAM 42 GiB).

Numerics: the pool computes with int8 activations, as it does in decode. Prompt-path logits (one-shot,
`bench/speed/tlppl.sh`): KL against T = 0 is 2.4e-3 (512 tokens) / 6.2e-3 (2,048), T = 0 twice 1.8e-3 / 5.9e-3 (the
prompt path is not run-to-run exact); top-1 99.0 / 98.2 % either way.

**In the service since 08:24** (`STRATA_XE_CPU_LIGHT=4` in `ds41-xe-engine.sh`, commit 4132422b; tpg.sh, then tsvc2):
needles found, the tool call right, svcbench 92.8 / 79.9 tok/s, essay 75.7 / 68.9, prompts 8K 754 / 882, 16K 1256, 31K
1602 tok/s, hidden RAM 42 GiB. The first build crashed at exit after a CPU-light prompt (close() freed a dense-card
buffer after the card list was cleared; seen in one-shot runs only) - fixed before the service took it.

Correction to the speed table below: the ~0.9 s `cache open` of a DSpark prompt is card 0's expert cache seeding again
after a batch of more than 4,096 tokens (1,690 pinned experts, 18 GiB at ~21.5 GB/s). Plain decoding pays the same
seed, after its prompt's own timer (it is in its time to first token, not in its prompt rate).

## Standard benchmarks (2026-10-24 ~05:30, and again ~10:15 after the night's prompt work: llama-bench's speed table, vLLM's serving benchmark)

Both on private copies of the service (`bench/speed/run_all.sh`, `run_all2.sh`: server + engine on port 18081 at the
service's settings, clocks pinned, 3 warm-up requests), once as the service runs (DSpark) and once without the draft
(`STRATA_SERVE_DSPARK=`: plain decode). The 05:30 run is the service before the night (down 05:06-05:30); the 10:15 run
has `STRATA_XE_CPU_LIGHT=4` and the prefetch thresholds (see the Measured section above).

**llama-bench's table** (`bench/speed/speedtab.py`): pp512 = tokens/s reading 512 new tokens, tg128 = tokens/s
generating 128, at context depth d; mean ± stddev of 5 runs after a warm-up run, as llama-bench prints them. Unlike
llama-bench: real text (Xolotl's own sources, then "Ignore the code above. Write ... essay"; random tokens route to other
experts and defeat the draft), through the API, rates from the engine's clock (the server's llama.cpp `timings`). The
engine resumes a conversation only where its last prompt ended, never inside a prompt, so depth d is one conversation
whose turns each add 512 tokens: the 5 runs sit at d - 1024 ... d + 1024.

| test | DSpark (the service) 05:30 | DSpark 10:15 | plain 05:30 | plain 10:15 |
| --- | ---: | ---: | ---: | ---: |
| pp512 @ d0 | 162.2 ± 7.1 | **238.1 ± 9.7** | 192.0 ± 5.4 | **283.4 ± 10.8** |
| pp512 @ d8192 | 157.7 ± 8.0 | **223.3 ± 10.7** | 183.4 ± 8.1 | **267.0 ± 11.5** |
| pp512 @ d32768 | 151.9 ± 4.2 | **215.2 ± 8.2** | 175.7 ± 5.1 | **252.0 ± 8.8** |
| pp512 @ d98304 | 135.6 ± 5.9 | **184.5 ± 5.8** | 155.9 ± 7.6 | **218.4 ± 7.3** |
| tg128 @ d0 | 72.0 ± 3.3 (2.13 tokens a pass) | 72.0 ± 5.2 (2.16) | 66.5 ± 0.5 | 66.8 ± 1.9 |
| tg128 @ d8192 | 74.1 ± 6.0 (2.21) | 73.9 ± 5.8 (2.18) | 67.6 ± 1.6 | 67.2 ± 0.6 |
| tg128 @ d32768 | 62.2 ± 7.2 (2.07) | 65.2 ± 4.3 (2.12) | 62.5 ± 0.9 | 62.2 ± 1.4 |
| tg128 @ d98304 | 62.4 ± 3.6 (2.17) | 64.4 ± 6.4 (2.26) | 61.1 ± 1.1 | 61.0 ± 1.4 |
| a first message from empty: 6,668 / 31,256 / 96,790 tokens | 10.0 / 23.2 / 62.5 s | 10.1 / 25.3 / 72.4 s * | 8.4 / 20.9 / 57.1 s | 8.7 / 20.4 / 56.9 s |

\* the first CPU-light build kept its sum in a ~70 MB buffer on card 0, which put DSpark's long prompts under
ggml-sycl's fp16 weight-cache floor; fixed at 10:55 (commit 7ef5fd41). With the fix (`bench/speed/ab8.sh`, 3 runs, the
same order, first messages of 7,170 / 31,758 / 97,294 tokens): 11.4 / 22.3 / 62.5 s (the pool off: 11.8 / 24.3 / 63.7).

**vLLM's serving benchmark** (`bench/speed/vbench.sh`: `vllm bench serve` 0.30 from the vllm-a image, no GPU; MT-Bench's
80 first turns, `philschmid/mt-bench`, vLLM's speculative-decoding set; 256 tokens at most, one request at a time,
thinking off, temperature 1.0 / top-p 0.95):

| | DSpark 05:30 | DSpark 10:15 | plain 05:30 | plain 10:15 |
| --- | ---: | ---: | ---: | ---: |
| time to first token, median / p90 / p99 | 0.80 / 2.10 / 3.52 s | **0.74 / 1.57 / 2.46 s** | 0.70 / 1.79 / 2.94 s | **0.70 / 1.40 / 2.24 s** |
| time per output token, median (mean) | 12.7 (13.2) ms = 79 tok/s | 13.0 (13.0) ms = 77 tok/s | 15.6 (16.0) ms = 64 tok/s | 15.8 (16.1) ms = 63 tok/s |
| inter-token latency, median / p90 / p99 | 0.1 / 38 / 52 ms (the draft's tokens come in bursts) | 0.1 / 38 / 51 ms | 15 / 19 / 29 ms | 16 / 19 / 29 ms |
| end to end, median | 3.77 s | 3.53 s | 4.30 s | 4.30 s |
| output tok/s over the run (prompts included) | 54.1 | 56.1 | 48.7 | 49.3 |

vLLM's client needed two fixes for this server (`bench/speed/patch_vllm_chat.py`, mounted over the copy in
`/opt/venv/lib/python3.12/site-packages` that the `vllm` command imports - not `/workspace/vllm`): the server sends the
OpenAI role chunk before it reads the prompt, and vLLM timed TTFT to it (~2 ms); and the client stripped each network
read before its SSE parser saw it, so a lone `: keep-alive` comment swallowed the events after it (the first words of
every answer). No vLLM warm-ups either: they repeat the first prompt, which the engine then answers from its snapshot
without reading it.

What it shows (05:30): a 512-token prompt cost ~3 s at any depth - it streamed the ~2,000-2,600 experts it uses that are
in no card's VRAM over PCIe (and hit ~7,000 that are); every agentic turn paid that. At 10:15 the light ones run on the
CPU pool instead: 512 tokens ~2.1-2.8 s (DSpark) / 1.8-2.3 s (plain) from depth 0 to 98K, MT-Bench's p99 time to first
token -30 %; decoding is unchanged. The draft gives +8-10 % on prose at
short contexts (tg128), +21-23 % on MT-Bench's mixed prompts (time per token) and nothing at 32K and beyond here, and
it costs prompt speed: -13-16 % on pp512 (the draft's weights take VRAM from the expert cache: ~20 % more experts
streamed a 512-token prompt, the experts phase 2.4 s instead of 2.15), -9-16 % on whole prompts as timed (DSpark's
also hold card 0's ~0.9 s cache seed after the batch, `cache open` in the engine log; plain pays that seed after its
prompt timer - corrected 08:25). Raw results: `bench/speed/out/`.

## AIME 2026 at DeepSeek's settings (2026-10-24 06:43-12:14, through the service)

DeepSeek's own eval (DeepInfer `eval/aime`: the prompt "{question}\nPlease reason step by step, and put your final
answer within \boxed{}." as one user turn, thinking at reasoning effort 100, temperature 1.0, top-p 1.0, 144K tokens of
prompt + output, no thinking budget; its rule judge: the last `\boxed{}` integer) through the service's OpenAI API,
one sample a problem where DeepSeek takes 32 (`bench/quality/aime.py`; the prompt is byte-identical to the
checkpoint's own encoder's, `encchk.py`; MathArena/aime_2026, DeepInfer's task ids).

| | |
| --- | --- |
| Pass@1, one sample a problem | **29 / 30 = 96.7 %** (DeepSeek's reference for the original model: 99.9 %, mean of 32 samples) |
| the miss | AIME I #15 (answer 83): thought for the whole 144K tokens without answering - coherent work (1,980 distinct lines among its last 2,000), not a loop; the server's loop recovery never fired |
| output tokens a problem | mean 22,068, median 14,942; AIME I #11 took 104,951 and was right |
| generation | 3.0 h, 60.6 tok/s mean (DSpark on math, ~2.2-2.3 tokens a verify pass) |

The run paused between problems for the night's A/B windows (a `STOP` file) and two problems ran while the engine
compiled (slower, same answers); the service's prompt changes landed in between - they move the prompt path's numerics
by the run-to-run floor only. GPQA Diamond (198 questions; DeepSeek 93.4 %, mean of 8 samples) runs the same way
(`bench/quality/gpqa.py`): started 12:15 on the idle service, it stops at the first other request (the client yields)
or 16:00 UTC; `gpqa.py --score` prints what it has, and starting it again resumes.

## Measured (2026-10-24 ~03:10: the CPU pool's threads placed, the draft's last stage fused)

The CPU pool (the experts in no card's VRAM, run on the host during a verify pass) is bound by memory bandwidth: a
layer's time is its experts' bytes at the pool's rate. Its kernels reach the DRAM's streaming ceiling for its threads
(~35 GB/s; `bench_cpu_pool`), but in the engine it ran at 21-28 GB/s: its 7 workers own CPUs 1-7, and the threads
that floated - the TP pass's host loop, the promotion helpers - landed on the workers' own cores. They are placed now:
the pass's host loop on CPU 0, 6 pool workers on cores 1-6, the helpers on core 7.

| one-shot DSpark, service settings | before | placed |
| --- | --- | --- |
| math: a one-expert CPU layer, the pool's rate | 543 us, 21.7 GB/s | 481 us, 24.5 GB/s |
| essay: the same | 523 us, 23.8 GB/s | 452 us, 27.4 GB/s |
| cycle at the common width mix (math / essay) | 36.25 / 27.77 ms | 35.66 / 27.04 (-1.6 / -2.6 %) |

The draft's Markov chain (its last stage: five draft slots one after the other, each a cross-card argmax) is fused:
6 commands a slot instead of 12, and the draft's cross-card sums go by store kernels rather than the copy engine the
promotions queue on. Same tokens and confidences as before (checked in the same graph on the same inputs); the draft
on the cards 2.36 -> 2.12 ms, essay cycle -1.8 % more.

**In the service since 03:10** (`STRATA_HOST_CPUS=0 STRATA_XE_POOL_WORKERS=6 STRATA_XE_HELPER_CPUS=7,15` in
`ds41-xe-engine.sh`; `tpg.sh`, then tsvc2): needles all found, the tool call right, svcbench 94.0 / 85.7 tok/s (85.8 /
83.9 at 01:55), essay 78.2 / 68.6 (76.5 / 70.4), prompts 8K 815 / 899, 16K 1270, 31K 1633 tok/s, hidden RAM 43 GiB peak.
Tried and dropped: the card repack again (the pool faster, but its kernels slow the draft: +3.5 ms a cycle), 16
promotions a card in flight (more copies beside the pool: math -6 %). Details: `engine1/DESIGN.md`.

## Measured (2026-10-24 ~01:55: DSpark's draft on the four cards)

A DSpark cycle was a verify pass on the four cards (engine1) plus the draft on card 0 alone: 4.9 ms a step on the
essay, ~17 % of the cycle - card 0 busy 3.65 ms of it, the rest the host submitting ~340 small ggml kernels. The
draft's three blocks are trunk layers (window attention only), so engine1 runs them now (`STRATA_TP_DRAFT=1`), split
over the cards like a trunk layer, its 128 Q2_K experts a block spread over the cards (a Q2_K form of the planar expert
kernels, its own kernels), one replayed graph a card. Its weights take ~1.2 GiB a card instead of 5 GB on card 0, whose
expert cache takes the difference. Both drafts on the same inputs pick the same token 92-98 % of the time a slot
(rounding differences). The draft step is ~1.9 ms of card time on every card.

The promotions had to change for it: their copies and repacks ran on cards 1-3 beside the old draft, but they held
every card's compute units at the start of this one (card 0 waited 3.5-5.5 ms at its first cross-card sum). The host now
writes the planes into the pinned bounce (a streaming repack, ~2 ms an expert), the cards only copy them in on their
copy engines, and the passes' / draft's inputs come through small kernels, not the copy engine the promotions queue on.

| one-shot DSpark, service settings, 1000 tokens | the ggml draft | the TP draft |
| --- | --- | --- |
| essay: cycle at the common width mix | 29.43 ms, 78.1 tok/s, draft 4.93 ms | 27.45-28.07 ms (-5..-7 %), 80.3-81.3 tok/s, draft 3.1 ms |
| code | 31.37 ms, 129.9 tok/s | 28.63 ms (-8.7 %), 148.1 tok/s, draft 2.5 ms |
| math | 37.6 ms, 107.8 tok/s | 36.5-38.6 ms, 102-110 tok/s (noisy: the CPU pool's text) |

**In the service since 01:55** (`STRATA_TP_DRAFT=1`; `tpg.sh` twice, then tsvc2): needles all found, the tool call right,
svcbench 85.8 / 83.9 tok/s (82.8 / 73.4 before), essay 76.5 / 70.4 (69.8 / 69.4), prompts 8K 761 / 881, 16K 1241, 31K
1598 tok/s, hidden RAM 42 GiB peak, card 0 1648 cached experts (1323). Also: `--serve`'s sampling candidates without
sorting the vocabulary (0.31 -> 0.08 ms a row, the same candidates). Details: `engine1/DESIGN.md`.

## Measured (2026-10-23 ~22:20: card 0's copies of the hottest experts, handed out every pass)

The cards waited ~4-9 ms a verify pass for each other, mostly because card 0 holds few of the hot experts in the later
layers and waits for the CPU pool in the middle ones. Card 0 now keeps copies of each layer's hottest expert-card
experts (2.5 GiB, uploaded at every cache seed while it keeps 1 GiB free), and `route_jobs` gives a pass's job of a
copied expert to whichever card finishes first - card 0 counted from its CPU-pool wait, during which it is free.
Which experts: the uses of three texts (`/opt/ai/dsv41/replica-usage.txt`); the calibration ranking predicts other
texts' hot experts poorly.

| one-shot DSpark, service settings, 1000 tokens | no copies | with copies |
| --- | --- | --- |
| essay cycle at the common width mix | 29.61 / 30.14 ms | 29.28 ms (-2 %) |
| code, 5-token verify passes | 26.38 ms | 25.49 ms (-3.4 %) |
| math cycle | 39.37 ms, 101 tok/s | 37.02 ms (-6 %), 110 tok/s |

The first service try crashed (see Safety); a full card now takes the copies back before an allocation fails.
**In the service since 22:20** (`tpg.sh` three times, then tsvc2): needles all found, the tool call right, svcbench
82.8 / 73.4 tok/s, essay 69.8 / 69.4, prompts 8K 831 / 931, 16K 1218, 31K 1648 tok/s, hidden RAM 43 GiB peak (the
copies' 2.5 GiB of pinned sources). Details: `engine1/DESIGN.md`.

## Measured (2026-10-23 ~20:35: the verify pass's own kernels, and the cards waiting for each other)

A verify pass keeps every card busy: per card the kernel times add up to the pass. Timed inside the pass graph (a
clock-reading kernel after every step, `STRATA_TP_STAMP_LAYER`), a 5-token layer is ~165 us of Q6_K mat-vecs, 120-190
us of routed experts, 55 us of attention, two 16 us all-reduce copies (PCIe-bound) and small steps. Three changes:

- `route_jobs` runs a lane a pick instead of one work-item (it was 4.5 us at 1 token, 15 at 5).
- The expert kernels stop at the pass's real job count and gate/up runs on 8 job slots, not one per possible pick
  (`STRATA_TP_NJ_DEV`, `STRATA_TP_JOB_GRID`; bit-identical).
- Each Q6_K mat-vec shape gets its fastest kernel form per pass width, timed at load on the card's own layers with a
  kernel between launches (`STRATA_TP_MV_TUNE`): XMX for q_b, wo_b and the shared expert's down from 3-5 tokens (32.7
  -> 24.6 us for q_b at 5), dp4a's 4-row form elsewhere. The XMX kernel now sums exactly (the scale after the MAD).

| one-shot, service settings, clocks pinned | before | now |
| --- | --- | --- |
| `tp_bench` 5-token pass | 22.96 ms | 22.06 ms (-3.9 %) |
| DSpark essay cycle at the common width mix | 30.39 ms (mean of 5 runs) | 29.70 / 30.17 ms |
| decode-path PPL, 5-token passes (w/o the chaotic row 111) | 2.0059 / 2.0037 | 2.0073 / 2.0087 (KL 0.0029, floor 0.0027) |

**In the service since 20:35** (code defaults; `tpg.sh` twice, then tsvc2): needles all found, the tool call right,
svcbench 85.5 / 75.1 tok/s, essay 70.9 / 66.8, prompts 8K 777 / 934, 16K 1318, 31K 1667 tok/s, hidden RAM 40 GiB peak.

What is left is waiting: every card idles ~4 ms of a ~24 ms pass at the FFN all-reduce on the essay (~6 ms on code,
~8.7 on math), and card 0 waits 2.2-2.8 ms for the CPU pool (4.7 code, 7.8 math). In layers 11-19 everyone waits for
card 0, which waits for the host; in layers 25-39 card 0 holds almost none of the hot experts and waits for the others.
A placement re-dealt from one text's expert counts does not carry to other texts (`bench/engine1/place.py`: -1.5 to
-2.9 ms on the other domains), so the next step is dynamic: copies of each layer's hottest experts on card 0, handed
to the least-loaded card each pass. Details: `engine1/DESIGN.md`.

## Measured (2026-10-23 ~17:30: the CPU-pool experts of verify passes - promotions in time for the next pass)

The plan was idle cards computing CPU-pool experts over PCIe. A per-expert log of the verify passes
(`STRATA_XE_CPU_LOG`) changed it. 94-95 % of the CPU pool's experts sit in the pageable part of the RAM arena: the
pinned ~20 GiB hold the hottest arena experts, and card 0's cache already has ~1,700 of those. The pool reads 28-31 GB/s
cold and 45 GB/s with its weights in the caches, against larry's 38-40 GB/s of DRAM. But 63 % of its experts were
repeats, 31 % of one the pass before had run, while a promotion to a card took 2-5 passes. So the promotions now land
before the next pass:

- A pageable expert's pinned bounce is filled right after the CPU pool ran it, while the pass goes on (streaming
  stores), not after the pass (`STRATA_TP_PC_EARLY`).
- Its copies (4 MB chunks under TP, were 1 MB) run during the draft, when cards 1-3 are idle.
- The TP core asks the tier before every pass, which publishes the promotions whose copies are done (`STRATA_TP_PC_PUBLISH`).
- A pass's Engram rows are read by a helper thread (`STRATA_TP_ENG_ASYNC`; neutral within the noise).

| one-shot, service settings, essay 1000 tokens, clocks pinned | before | now |
| --- | --- | --- |
| CPU-pool experts a verify pass | 10.2 / 11.3 | 7.2 / 8.0 |
| DSpark cycle (verify + draft) at the same pass-width mix | 31.5 / 32.6 ms | 30.1 / 30.2 ms (-6 %) |

That is ~+6.5 % tokens a second at equal acceptance; tok/s itself moves with the text, which diverges after a few
tokens. Promoting at the first use was worse (more evictions, the draft slower), and so was pausing the fills while
the pool computes (`STRATA_TP_PC_PACE`). Every published slot was checked against a fresh repack once (1392, 0 bad).
**In the service since 17:31** (code defaults; `tpg.sh` twice, then tsvc2): needles all found, the tool call right,
essay 69.9 / 63.6 tok/s, svcbench 80.4 / 72.9, prompts 8K 816 / 932, 16K 1303, 31K 1673 tok/s, hidden RAM 40 GiB peak.
The service samples at temperature 1, so its acceptance noise is larger than the gain. Details: `engine1/DESIGN.md`.

## Measured (2026-10-23 ~06:00: engine1's pager, in the service)

With TP on, only the prompt path reads card 0's full copies of q_b, wo_a, wo_b, the shared expert and the Engram
projection (4.93 GiB); the other cards hold the same weights in their shards. `STRATA_TP_PAGE=1`
(`tp/tp_pager.cpp`) keeps them on card 0 as one unit of virtual memory a layer. It gives the pages back after load
and maps a layer into one of 3 slots (244 MiB) when the prompt path reaches it, filled from the shards while the
layer before computes. Card 0's expert cache gets the freed VRAM: 14.2-16.9 GiB of experts in the service instead
of 10.0-11.3. Prompt passes of up to 7 tokens run on the TP core.

| one-shot, service settings, clocks pinned | TP, pager off | pager on |
| --- | --- | --- |
| essay DSpark | 63.0 tok/s | 69.6 / 69.0 / 71.1 |
| 95K prefix, DSpark | 50.3 tok/s | 57.0 |
| plain decode | 61.6 tok/s | 63.5 |

**The first deploy (05:20) garbled long prompts** and was reverted five minutes later. The 8K / 16K / 31K needles
came back as word salad, and so did the tool call after them. The fills were the other cards' copies into card 0's
virtual memory, and once card 0 remaps an address to new pages, another card's copy into it still lands in the old
pages (`bench/engine1/vmm_remap.cpp` shows it in 30 lines, every time). So a slot moving to another layer left
cards 1-3 writing over other layers' weights and over the expert cache that had taken the freed VRAM. A
private-port copy of the service (`bench/engine1/tpg.sh`) reproduced it exactly and cleared the other suspects
(VRAM pressure, cross-card barriers). Card 0 now pulls every fill on its own copy queue. The refill check is
bit-identical, and the service sequence ran twice in one server without a fault.

**Row 111 is not a race.** One p2k prompt row (token 1018 in a repeated pattern; the next token is certain)
predicts the wrong token in about one run in four: NLL 7 instead of 0.004, PPL +0.006. It does so with the pager off
too (2 of 8 runs). `STRATA_DUMP_COL=111` dumps that row at every layer. Its routed sums match F32 reference experts
computed from the GGUF (`bench/engine1/test_ds41_dumpref.cpp`) and its routing matches the router recomputed from
the dump (`routercheck.py`). The row is chaotic: good runs already differ by 30-80 % inside, with the same answer,
and the engine's ordinary run-to-run noise decides which way it falls. The bulk numbers don't move with the pager.

**In the service since 2026-10-23 06:02** (`STRATA_TP_PAGE=1` in `ds41-xe-engine.sh`; 0 = card 0 keeps the copies).
`tsvc2.sh`:

| tsvc2.sh check | `STRATA_TP=4`, pager off | + pager |
| --- | --- | --- |
| needles (200 / 8K / 8K / 16K / 31K tokens) | all found | all found |
| essay | 65.1 / 63.4 tok/s | 69.5 / 66.0 |
| svcbench | 80.4 / 81.6 tok/s | 80.3 / 76.2 (its noise) |
| prompts 8K | 786 / 942 tok/s | 795 / 927 |
| prompts 16K / 31K | 1342 / 1689 tok/s | 1162, then 1419 / 1437 / 1646 |
| short prompts (26-34 tokens) | 30-40 tok/s | 52-78 (TP passes) |
| tool call | correct | correct |
| hidden RAM | 40 GiB peak | 40 GiB peak, 21 GiB available at the lowest |

## Measured (2026-10-23 ~02:30: engine1 at the service's settings)

The FLEX 3 failure below came from the expert settings, not from TP. FLEX 3 GiB with a 0.3 GiB margin left the expert
cards ~0.3 GiB free, and a 252-token prompt ran out of VRAM in the tier's chunk path. Prompts below 2048 tokens take no
prefetch, which is the path that resizes the flex region. At the service's own settings (ctx 262144, FLEX 1 / margin
1.5, RESERVE 3.5), eager TP was DSpark -2 % and plain +15 %, for two reasons:

- **VRAM.** Its KV replicas displaced ~960 VRAM experts, doubling CPU-pool uses.
- **Host CPU.** Its four normal-priority promotion helpers spend 10-20 ms of host repack on each expert. They slowed
  the draft's kernel submission (3.95 vs 2.30 ms) and the verify pass (one helper: 34.1 -> 30.7 ms).

Fixed:

- `STRATA_TP_LAZY_KV`: ranks 1..N-1 keep the KV replicas in Level Zero virtual memory, one address for the graphs.
  Pages are mapped as a pass's cap bucket or a sync reaches them, and the new rows are copied from card 0. Until then
  that VRAM is lent to the tier's flex region; `tp_make_room` takes back only the shortfall.
- `STRATA_TP_PC_DEVREPACK`: promotions copy the file layout, pinned arena blobs straight from the arena, and the card
  repacks. Two helpers.
- The head split reads TP's row slices, and the expert cards' last 1 GiB slab is filled at placement
  (`STRATA_XE_FILL_SLAB`, both paths: today's path +0.74 GiB of experts a card, CPU-pool uses -10 %).

| one-shot, service settings, clocks pinned, both with the slab fill | today | `STRATA_TP=4` |
| --- | --- | --- |
| essay DSpark | 60.5 / 56.9 tok/s (verify 33.7 / 33.3 ms) | 61.2 / 64.7 (28.7 / 28.8) |
| 20K prefix, DSpark | 92.0 tok/s (verify 37.9 ms) | 110.0 (34.1) |
| 95K prefix, DSpark | 50.4 tok/s (verify 40.5 ms) | 50.3 (38.7; 2.25 vs 2.33 tokens a pass) |
| plain decode | 47.1 tok/s | 61.6 |
| prompts (20K / 95K) | 824 / 1445 tok/s | 777 / 1490 |

Numerics: decode-path PPL (p2k, exact routing) 2.0085. KL against the eager-replica TP run is 0.0033, TP's
run-to-run floor. All promoted slots checked against a fresh repack: 0 bad.

**In the service since 2026-10-23 02:35** (`STRATA_TP=4` in `ds41-xe-engine.sh`, FLEX 1 / margin 1.5 unchanged;
`STRATA_TP=0` brings back the old path). `tsvc2.sh` against the service as it was before (55.6 tok/s essay after the
FLEX 3 revert):

| tsvc2.sh check | before (2026-10-22 evening) | `STRATA_TP=4` |
| --- | --- | --- |
| needles (200 / 8K / 8K / 16K / 31K tokens) | found | all found |
| essay | 55.4 / 54.9 tok/s | 64.7 / 61.1 (+16 %) |
| svcbench | 72.9 / 67.2 tok/s | 76.5 / 73.3 |
| prompts 16K / 31K | 1292 / 1611 tok/s | 1333 / 1692 |
| prompt 8K | 976 tok/s | 815 / 944 (the first one cold) |
| tool call | correct | correct |
| hidden RAM | 41 GiB peak | 40 GiB peak (limit 54), 21 GiB available at the lowest |

## Measured (2026-10-23 early hours: engine1, decode / verify on N cards, tensor parallel - built, not in the service)

`STRATA_TP=N` (default 0 = the path above) runs every decode / verify pass of <= 7 tokens on a new core,
`xolotl/tools/v41/engine/tp/` (`libds41_tp.so`, design and status in its `DESIGN.md`): the dense half split over N
cards - nothing assumes 4, the splits come from N and the shapes (N = 3: o-groups 3 / 3 / 2) - and the host out of
the loop (each card replays one SYCL graph a pass).  Rank k computes q_b / attention / wo_a for its o-groups, wo_b's
matching K columns, its n_ff blocks of the shared expert, its rows of the Engram projection and the LM head, and the
routed experts it holds.  Every card runs the rest (hc mixes, norms, q_a + kv, the window ring, the compressor, the
indexer, the router) on the same bits, so they all get the same results.  The cards meet twice a layer: copy-engine
card-to-card copies plus sequence flags, with bounded spins.  The consumer adds the N partials in rank order, so the
replicated state stays bit-identical.  Rank 0 adds the CPU pool's share; it waits for it only on layers whose
routing has one.  The prompt path, DSpark's draft and park / snapshot are unchanged: rank 0 works on the dense half's
own state tensors, and cards 1..N-1 pull copies after anything else writes them.

| bench (`tp_bench`: synthetic state, all experts in VRAM) | N = 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| decode pass (1 token, 40 layers) | 23.1 ms | 16.9 | 14.8 | 13.3 |
| verify pass (4 tokens) | 43.7 ms | 30.4 | 27.7 | 23.3 |

In the engine (one-shot essay x400, bias 0.5 / 0.3, `STRATA_XE_FLEX_GIB=3` on both sides, clocks pinned). The speed
A/B ran on the build just before rank 0 started reading the engine's weights in place, which frees 1.1 GB more of
card 0 for experts; it has not been re-timed since. The PPL and 95K runs below used the final build.

| | today | `STRATA_TP=4` | |
| --- | --- | --- | --- |
| plain decode | 45.0 / 45.5 tok/s | 60.5 / 60.6 | +34 % |
| DSpark | 59.95 / 59.11 / 60.06 tok/s | 61.53 / 60.62 / 65.66 | +5 % |
| DSpark verify pass | 32.7 / 33.7 / 33.5 ms | 30.3 / 30.1 / 27.6 | -12 % |
| DSpark draft pass | 5.1 / 5.0 / 5.2 ms | 6.3 / 6.2 / 6.0 | +1.1 ms, not explained yet |
| 95K prefix (ctx 131072): prompt / DSpark | 1408 tok/s / 48.6 | 1447 / 49.2 | neutral |

Where a TP pass goes (`STRATA_TP_PROF=1`):

- **Plain pass at steady state: 15.0 ms.** Waiting on the cards takes 11.9 ms (the host's polls of the per-layer
  mailboxes), the CPU-pool experts 1.3, Engram 1.0, and inputs, launch and the final card wait 0.8.
- **Verify pass: 26-30 ms.** The cards take 15.7-17.5 ms and the CPU-pool experts 7.2-8.9 (10.7-11.3 layers a pass
  have one).

Correctness, from decode-path logits on p2k with exact routing:

| | today | TP |
| --- | --- | --- |
| PPL, two runs | 2.0142 / 2.0243 | 2.0051 / 2.0059 |
| KL between its own two runs | 0.0068 | 0.0030 |

KL today-TP is 0.015-0.021 (top-1 agreement 97.9-98.3 %). TP's split sums round differently, so KL is 2-3x today's
run-to-run noise; PPL is the same or lower.

**Why it is not in the service.** At the service's ctx 262144, `STRATA_TP=4` with FLEX 3 failed the first long prompt
("prefill failed: tier run refused", then `UR_RESULT_ERROR_OUT_OF_RESOURCES`). TP needs ~2 GB of weights a card,
even with rank 0 reading the engine's planar weights in place (-1.1 GB on card 0), plus KV replicas sized for 262K
(~0.85 GB a card; ranks 1-3 at 2.75-2.87 GiB). Together with the bigger flex, that left the prompt path's buffers no
room. The service is back on `STRATA_TP=0` / FLEX 1 (essay 55.6 tok/s, needle found). The same VRAM is what limits
every run above: it displaces ~600 VRAM experts, so TP runs make 1.4-2.6K CPU-pool expert uses against 1.1-1.6K.

Lessons, all measured:

- Host-to-GPU flags need an acquire fence (system scope) plus a volatile load on every poll. Atomic acquire loads never
  see the host's write (`bench/engine1/hs_probe2.cpp`).
- A second in-order queue does run while the card's first queue spins: 16 MB H2D in ~710 us
  (`bench/engine1/cq_probe.cpp`). The promotion copies therefore run beside the passes.
- The first card-to-card copy of each pair sets up the path (~6 ms), so init does it before any cross-card wait.
- Stacked Q6_K weights must be planarized as one plane set.
- The promotion cache needs several jobs a card in flight, set at every call: DSpark's 16-token seed passes run the
  tier's own `pc_setup` first.
- Recording a graph costs ~40 ms a width. All 14 graphs are pre-recorded at attach, with a cap floor of 1024 rows.

## Measured (2026-10-22 evening: fused decode kernels)

A decode / verify layer's small-op chains on card 0 now run as fused ggml-sycl ops (larry's ggml patch: the kernels in
`ggml-sycl/dsv41-dec.hpp`, pointer-based so a future engine can call them; wrappers in `dsv41-fused.cpp`).  Every switch
below defaults on; `0` brings back the op chain it replaces:

| switch | what runs fused | replaces (per decode layer) |
| --- | --- | --- |
| `STRATA_ATTN_FUSE` | `kv_fin`: the window row's rms_norm * w, RoPE, FP8 rounding, its ring write and the pass's RoPE table; `attn_dec`: the window + top-K compressed keys gathered straight into local memory, masks, q RoPE / o de-RoPE by the table, sinks | rope q, rms+rope+fq+set_rows kv, concat, kgather2, concat + cast mask, rope back o |
| `STRATA_Q8_FUSE` | the ops in front of the Q6_K mat-vecs write the int8 activations fastmv reads (hc_fb, q_a's norm, the attention, the shared expert's SwiGLU) | 5 quant_x + wo_a's cont |
| `STRATA_DIRECT_OUT` | the FFN hc and the router write the layer's state tensors in place | the batched copy, the ids' float -> int copy, the shared expert's copy |
| `STRATA_LI_FUSE` | the lightning indexer from its two raw products (q RoPE + FP4 rounding, weight scale, visibility / candidate masks, the candidate blocks' maxima) and its top-k sorted straight into the selection | ~14 on each of the 8 indexer layers |
| `STRATA_HC_MERGE` | each hc_fa + hc_fb pair as one kernel (the token's last work-group runs hc_fb's part) | 2 |
| `STRATA_FIN_DEFER` | the finish (hc_post2) inside the next layer's attention hc; state readers flush it | 1 + a graph |

Correctness: `test_ds41_attnfuse` (GPU, `GGML_SYCL_LI_XMX=1`) runs every fused op against the chain it replaces - all
bit-identical except the attention output (1 ulp in ~1.4 % of values: the q rotation from the table contracts its
multiply-adds differently from ggml's rope kernel).  Decode-path logits (p2k, exact routing) vs the unfused path: KL
0.009, top-1 98.1 % - the same as two unfused runs against each other (KL 0.009, 98.4 %): **the decode path is not
run-to-run deterministic**, even unfused and with exact routing (same config: PPL 2.0232, 2.0232, 2.0023; the morning's
"bit-identical twice" was two runs that happened to agree).  The unit test also found a latent bug, fixed: the oproj
op's own (non-hook) path read every wo_a slice past the first out of bounds (it used the 2-D parent's nb[2] as the
slice stride; only the tier's hook had ever run that op).

Speed (one-shot essay, service env, bias 0.5, clocks pinned):

| | all off | fused | |
| --- | --- | --- | --- |
| plain decode | 42.2 / 42.4 tok/s | 44.9 / 46.6 (+ hc merge 46.6 / 46.3) | +8-10 % |
| DSpark verify | 38.9 / 38.7 / 37.0 ms/pass | 34.8 / 34.2 / 33.8 | -10 % |
| host submit a layer (attention / shared expert) | 212 / 37 us | 129 / 19 us | |
| GPU timeline a layer (attention graph / shexp / expert-sum wait / finish) | 395 / 82 / 87 / 9 us | 357 / 74 / 69 / 2 us | 603 -> 536 us |

Through the service (`tsvc2.sh`): svcbench 72.9 / 67.2 tok/s, essay 55.4 / 54.9 (was 69.4 / 66.7 and 52.7 / 49.7),
prompts unchanged (8K 976, 16K 1292, 31K 1611 tok/s), needles and tool calls fine, hidden RAM peak 41 GiB.

Lessons: transcendentals per work-group are not free - the first `attn_dec` recomputed the RoPE angles in every
work-group and every (query, head) and took 58 us against the old path's 26; one table per pass (computed by `kv_fin`,
which needs the same angles) plus the chunk's key-row addresses staged in local memory brought it to 22.7 + 5.0 us
(`bench/fuse/bench_attn.cpp` times the kernels in isolation).  Where the layer's time goes now (the timeline above):
the expert cards finish ~80 us after the router readback, but card 0 runs the shared expert (74 us) and then its own
expert jobs (61 us, 38 % of calls) before the sum - the next lever is the shared expert / card 0's expert jobs on the
expert cards, not more dense-side fusion.

## Measured (2026-10-22 late morning: the routing bias, the quant against the original, the new engine's step 0)

**The decode routing bias** (`--route-bias X --route-bias-file Y` in `larry-ds41.json`; service 0.5 / 0.3, engine
default 1.0 / 0.6 - what every one-shot run without the flags used).  It is selection-only: during decode / verify
passes the router adds +X to the experts resident in VRAM before its top-6 (the mixing weights stay the model's), so
some tokens are computed by a different expert than the model picked; prompts always route exactly.  Larry has no
NVMe-tier experts, so the file penalty never applies here.  One-shot, the service's env, GPU clocks pinned
(`qv/tbias.sh`, `qv/tbias2.sh`):

| bias | DSpark essay tok/s | plain tok/s | VRAM hits | KL vs exact (decode path, 2K tokens) | top-1 = exact's | ppl (p2k) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (exact) | 37.9 (39.0 / 37.6 / 37.0) | 32.4 | 89 % | 0 (two runs bit-identical) | 100 % | 2.010 |
| 0.25 / 0.15 | 46.7 (46.4 / 47.0) | - | 96 % | 0.0145 | 97.8 % | 2.009 |
| 0.5 / 0.3 (service) | 53.4 (55.2 / 53.0 / 52.1) | 42.8 | 98 % | 0.0179 | 97.1 % | 2.003 |
| 1.0 / 0.6 (default) | 57.6 (56.0 / 59.2) | - | 99.4 % | 0.041 | 95.6 % | 1.989 |

Exact routing costs ~29 % here: ~10 % of the expert picks land on the RAM arena's CPU pool (verify passes 57-59 ms
vs 37 ms).  The bias lowers the perplexity of this text a little (the favoured hot experts were ranked on an
agentic / coding census) - whether it moves the model away from the original is what the teacher comparison below
answers.

**The quant against the original model** (`qv/`, files in the mirror's `qv/`): the ORIGINAL DeepSeek-V4.1-Flash-1017
(FP8 dense, FP4 experts, as DeepSeek ships it) teacher-forced on larry's CPU - DeepSeek's own `inference/model.py` on
Xolotl's torch stand-in kernels (an independent review found them faithful: same rounding / scales / nibble order,
only accumulation order differs), one layer at a time through Xolotl's `v41_calib.py` with each 7.4 GB shard
downloaded, sha256-checked and deleted after its layer (298 GB streamed; the Engram tables are the engine's extracted
files, their small projections range-fetched); 40 layers in ~3 h.  Then our engine (the service's settings) over the
same 18 x 2048 tokens - English / Chinese Wikipedia, MATH-500 + GSM8K solutions, SWE-agent trajectories as chats, code
(this week's larry C++, DeepSeek's model.py, CPython's argparse) - full logits per row, scored by `qv_score.py`:

| domain | original ppl | our ppl | ratio | mean KL(orig ‖ ours) | median KL | top-1 = original's |
| --- | --- | --- | --- | --- | --- | --- |
| code | 1.93 | 2.05 | 1.06 | 0.10 | 0.002 | 92.0 % |
| math | 1.66 | 1.79 | 1.08 | 0.16 | 0.004 | 92.0 % |
| agent | 2.85 | 3.17 | 1.11 | 0.32 | 0.011 | 86.7 % |
| English | 7.08 | 8.12 | 1.15 | 0.31 | 0.13 | 78.2 % |
| Chinese | 4.51 | 6.51 | 1.44 | 0.60 | 0.08 | 75.4 % |
| all (36,846 tokens) | 3.16 | 3.63 | 1.15 | 0.29 | 0.022 | 85.0 % |

(prompt path, exact routing - how the service reads every prompt.)  Generation (decode path, 10 of the sequences):
exact routing ratio 1.12 / KL 0.27 / top-1 84.1 %, the service's bias 0.5 1.16 / 0.32 / 83.0 % - the bias's extra
cost sits in Chinese (1.30 -> 1.46) and English (1.13 -> 1.18); code (1.077 / 1.077), agent (1.05 / 1.06) and math
(1.07 / 1.07) barely move.  So: the full model, not broken, but clearly a step down from DeepSeek's release - about a
2-2.5-bit-quant-level loss overall (most tokens agree; a tail of confident disagreements, p99 KL 1.5-6), best kept on
code and math (the calibration's target), worst on Chinese.  The publisher's own evidence (HumanEval 21/21, tool
calls 10/10) is consistent with that but measures no fidelity; its "full evaluation" was still pending.

**The new engine, step 0** (`bench/engine0/tp_layer.cpp`): one decode layer rebuilt from the real fastmv mat-vecs on
real-size weights (cold, 40 layers), delay kernels of the traced small ops, 22 us a routed expert and real
card-to-card sums (13.3 us an all-reduce of 5120 floats), timed with the host out of the loop (all commands queued
behind a gate) and as recorded SYCL graphs (cross-card sync: sequence numbers copied into the peers' flags + bounded
spin kernels):

| us a layer | today's layout | dense half over 4 cards |
| --- | --- | --- |
| host submitting live | 540 (the real engine: 564) | 925 |
| host-free, queued | 532 | 410 |
| graph replay | 504 | 378 |
| graph replay, small ops in fewer launches (same summed time) | 474 | 355 |

The 300-us gate is not met: the split buys ~1.5-1.6x and only without the host in the loop (live, the host cannot
feed four cards).  Dependent kernels start ~0.5 us apart either way (1.49 / 1.54 us for 1-us kernels, queued /
replayed).  What is left is the ~45 tiny ops a layer (~108 us of kernel time on every card): fused for real (their
work is tiny; most of their time is per-kernel latency) they would plausibly bring the split layer to ~270 us and
today's to ~420 - to be measured by writing them.

## Measured (2026-10-22: long contexts, expert kernels per job, q_a + kv, routing on the GPU)

Service after this round (`tsvc2.sh`, commit 7116d9fa; `ds41-xe-engine.sh` now sets `GGML_SYCL_LI_GEMM_MIN=1`,
`GGML_SYCL_LI_XMX=1`, `STRATA_XE_JOBNT=1`, `STRATA_XE_COMBINE=1`, `STRATA_QKV_FUSE=1`):

| | before this round | now |
| --- | --- | --- |
| svcbench | 64.5 / 63.8 tok/s | 69.4 / 66.7 |
| essay (3 runs) | 53.3 / 48.4 | 52.7 / 49.7 |
| the essay behind a 97K-token prefix (`bench/gr/ctxsvc.py`) | 24-31 (the service's log at 90-100K) | 46.9 cold, 44.6 / 43.8 / 46.1 prefix-cached |
| prompt 8K / 16K / 31K / 97K | 963 / 1221 / 1510 / - | 965 / 1229 / 1600 / 1544 tok/s |

Needles 200-31K found, tools fine, hidden RAM peak 41 GiB (available >= 29 GiB).

**Long contexts were slow, and only there.** The service's own log (agentic turns with 76-100K-token, mostly
prefix-cached prompts) showed decode at 24-31 tok/s at 90-100K context against 42-49 at <= 36K. Cause: the lightning
indexer of every decode / verify pass ran ggml-sycl's per-row kernel (batches below `GGML_SYCL_LI_GEMM_MIN` = 32) at
~0.35 TFLOPS, ~23 ns a key row per query: at 98K ~2.3 ms an indexer layer per query, 8 indexer layers (2, 8, 14 at
ratio 2; 20-36 at ratio 1). One-shot, DSpark, GPU clocks pinned, a 95K-token context:

| | verify pass | decode |
| --- | --- | --- |
| before | 115.1 / 115.6 ms | 21.6 / 21.6 tok/s |
| `GGML_SYCL_LI_GEMM_MIN=1` (the prompt path's XMX GEMM for every batch) | 57.4 / 59.6 ms | 39.6-60.3 (3.0-4.0 tokens a pass) |
| + `GGML_SYCL_LI_XMX=1` (new: one fused bf16 XMX kernel for <= 8 queries) | 51.8 / 51.6 ms | 44.0 at 3.0 tokens a pass |

The same essay request with and without a 95K prefix (`bench/gr/tctx.sh`): 53.1 / 55.5 tok/s vs 61.2 / 55.6 - a long
context now costs ~2-4 ms a pass (unitrace: li_xmx ~13 + top-k ~8 + masks ~5 us a layer). Keys and queries are FP4
grid values stored as BF16, so bf16 XMX is exact for them (`_CHECK=1`: <= 2.4e-7 from li_gemm); short contexts
unchanged; decode-path PPL 1.9805 / 1.9911 / 1.9805 (before) vs 1.9909 / 1.9850 / 1.9888 (noise).  Verify ms a pass
only compare at equal pass widths: the repeated test prompt drives DSpark to 3-4 tokens a pass.

- **`STRATA_XE_JOBNT=1`**: the armed decode / verify sequences picked each expert kernel's form from the pass width;
  now each job's own token count picks it (`test_ds41_xe --bench-q8`, new bit-exactness column): 1-token jobs in the
  4 / 8-token forms 78.4 / 87.1 -> 71.2 / 72.0 us, 2-token jobs in the 8-form 106.8 -> 99.3; all forms bit-identical
  (=2, separate gate/up launches per width, was slower).
- **`STRATA_QKV_FUSE=1`**: each layer's q_a and kv (both read the attention norm's output) as one 5120 x 1792 weight
  over their bytes - the loader creates attn_kv right after attn_q_a so the allocator puts them back to back. Two runs
  each: plain decode 44.74 / 44.33 vs 44.70 / 44.50 tok/s (flat), DSpark verify 35.6 / 34.5 vs 33.2 / 34.4 ms a pass
  (-3.6 %), 31.6K prompt 22.44 -> 21.45 s (1411 -> 1476 tok/s); PPL prompt path 2.0036 -> 2.0020, decode path
  1.9888 -> 1.9850 (noise). Bench (`bench/gr/tmv.sh`, cold): q_a 18.8 + kv 8.9 us -> 24.0 at one column,
  29.1 + 13.4 -> 31.3 at five.
- **`STRATA_CONT_SKIP`** (default 1): no ggml_cont (a copy op) of fake-quant inputs or the compressed rows' mask that
  are contiguous already - two copies a decode layer: plain decode 43.94 / 44.32 -> 44.74 / 44.65 tok/s (+1.3 %), same
  values.
- **`STRATA_XE_COMBINE=1`** (commit 99eea657): prompt chunks - each expert card pushes its part to the dense card when
  it is done: 31.6K 23.0 -> 21.5 s, exact.
- **`STRATA_XE_GROUTE`** (off): the routing moved onto the dense card - a route kernel writes each card's job list and
  the CPU part's header (pinned), the expert cards are released by its event and pull their jobs card-to-card, card 0's
  share runs on the dense queue; the host only runs the CPU part and queues the sum (`_CHECK=1` compares with the host's
  grouping: 0 mismatches over thousands of calls). The timeline moves (verify layer 882 -> 844 us, host after the
  readback 149 -> 84 us), the tokens do not: plain 44.3 / 44.8 vs 44.2 / 44.3 tok/s, DSpark verify -0.6 ms a pass
  (noise) - the sum still waits for card 0's own experts and the CPU part. With it, `STRATA_TP_OPROJ=2` plain +2.5 %,
  DSpark neutral; `STRATA_HEAD_SPLIT_MB=30` -14 % (and once 1.000 draft tokens a pass); `STRATA_XE_ARM_GRAPH` (each
  card's armed sequence replayed as a recorded graph) -1 %. None of them for the service.
- **Host-signalled events** (`bench/gr/hs_probe.cpp`, the precondition for a host that queues layers ahead): a Level Zero
  event signalled by the host (`zeEventHostSignal`) works as a SYCL dependency (`make_event`): 2.9-6.2 us from the
  signal to the dependent kernel's write, and the host keeps queueing meanwhile (2000 kernels in 5 ms behind the
  waiting one); the first such submit costs ~1.7 ms.  A spin kernel's system-scope atomic loads of a malloc_host flag
  saw the host's write after 2.9 us here - unlike the 2026-10-21 spin probe, which ran to its bound (not re-checked
  why); loads through an uncached annotated_ptr and fetch_add(0) never saw it.
- **fastmv split-K knobs** (`GGML_SYCL_FASTMV_SG` 4096, `_UPL` 2: the sub-groups a launch should reach, the units of 32
  each lane keeps at least; `bench/gr/tmv.sh`, cold = weights rotated past the L2): fewer K splits help the 5120-wide
  matrices of <= 1792 rows and the F16 router - `_UPL=3`: q_a + kv 24.0 -> 20.5 us at one column, 31.3 -> 25.9 at
  five; router 10.7 -> 10.2 / 20.2 -> 17.1 - but cost the shared expert at 4-5 columns (gate+up 41.4 -> 45.1, down
  41.8 -> 43.7); `_SG` 2048 / 8192 trade the same way.  Not switched: ~4 us a layer (< 1 %), below the end-to-end
  noise, and it moves shapes the bench does not cover (the indexer's q projection) - a per-shape rule is the way.
- Where a 31.6K prompt's card-0 time goes (trace): copies 3.6 s (expert uploads to card 0, the X sends at ~35 GB/s),
  the indexer ~3 %.

## Measured (2026-10-21 night: where the four cards' time goes, and spreading work over them)

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

## Measured (2026-10-21 evening: prompt stalls, the DSpark draft, the verify expert phase)

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

## Measured (2026-10-21 afternoon: fused hyper-connections, and where the time goes now)

One-shot, the essay prompt (`pe.i32 -n 400`), service settings (`STRATA_D3_PF=0`):

| | before | now |
| --- | --- | --- |
| plain decode | 41.3-41.9 tok/s | 43.2-43.9 tok/s (+5 %) |
| DSpark verify pass | 41.4-42.8 ms | 38.8-39.8 ms (acceptance 2.3-2.6 tokens a pass, run to run) |
| decode-path PPL (2K tokens) | 1.9724 | 1.9775 (noise) |

- **Fused hyper-connection ops** (`STRATA_HC_FUSE`, commit 99ca748f; DeepInfer's "mega mhc" idea): decode / verify run each
  sub-layer's hc step as two ggml ops (`GGML_OP_DSV41_HC_FA`: hc_post + hc_pre + partial mixes dots / sums of squares
  per 128 columns; `_HC_FB`: the reduction, hc_mixes, the Sinkhorn on sub-group shuffles, the RMS norm) and the finish
  as one (`_HC_POST2`): 14 tiny kernels a layer -> 5. `test_ds41_hcfuse --gpu`: worst rel diff 4.6e-7 vs the composite.
- ggml-sycl's int8 activation cache keeps 4 inputs (kv re-quantised attn_norm(x) after q_b evicted it): neutral.

The decode layer now (`STRATA_GPU_TL=1`, real clock): attention graph 408 us, shared expert 84, then 62 us until the
routed sum is done, finish 4: 560 us. A verify layer (5 tokens): attention 557, shared expert 108, the sum's
dependencies met 249 us later, sum 9. **Open problem** (resolved for verify passes in the evening section: real expert work, not a lag): the dense queue sees the expert cards' parts 25-30 us (decode)
/ 60-110 us (verify) after they landed, although the host queues the sum ~30 us after the readback and the same
dependency costs 2-8 us in every probe (`bench/dqb/sum_lat2.cpp`, `p2p_lat.cpp`, `hostev.cpp`). Tried and reverted:
a one-lane flag wait on card 0, a host-event bridge thread, the dense card's experts on the dense queue, a host event
for the CPU part (the CPU-pool experts cost ~40 us a verify layer: `STRATA_XE_CPU_SKIP` 39.6 -> 37.4 ms/pass).

The prompt path is bound by host memory: an 8.5K prompt streams 62 GB of RAM-arena experts per batch, a 2K prompt the
whole arena (70 GB) - 61 % of those experts serve <= 4 tokens. Host -> cards DMA tops out at 28.4 GB/s for one card
or all four (one x16 uplink, `bench/dqb/h2d_bw.cpp`), but the prefetch averages 11-12 GB/s: the pageable 70 % of the
arena is staged into pinned buffers by 4 threads at ~3.5 GB/s each, and those copies plus the DMAs' reads saturate
DDR4. Measured without gain: non-temporal staging stores (19 GB/s for one thread alone, nothing inside a prompt),
transparent huge pages on the arena (2.6 of 48 GiB became huge pages; slower), 6 / 8 stager threads (10.7 / 11.4 s
vs 10.3-10.8 at 4: the copies share ~14-17 GB/s whatever the thread count),
the CPU pool for light experts (2K prompt +6 % at best, 8.5K slower, int8 activations move the greedy output).
**DANGER**: registering the pageable arena for device copies (`prepare_for_device_copy` = a Level Zero host-pointer
import) took card 0's VM down (`xe ... VM worker error: -16`, `UR_RESULT_ERROR_DEVICE_LOST`) - do not.

## Measured (2026-10-21, the utilization round)

| service (public endpoint, server timings) | goal start (2026-10-19 evening) | after the doubling round | now |
| --- | --- | --- | --- |
| `svcbench.py` (4 short answers, DSpark) | 20.6 tok/s | 50.3-55.6 tok/s | 57.4-60.0 tok/s (2.8-2.9x) |
| `essay.py` (800 tokens, mean of 3) | 17.3 tok/s | 38.5-42.6 tok/s | 44.1-44.8 tok/s (2.6x) |
| prompt ~8K tokens (`needle.py 8000 1`, cold) | ~310 tok/s | 666-717 tok/s | 883-914 tok/s (2.9x) |
| prompt ~16K tokens | - | 832-918 tok/s | 1042 tok/s |
| prompt ~31-32K tokens (needle found) | ~440 tok/s | 946-1114 tok/s | 1208-1223 tok/s (2.8x) |
| driver-held ("hidden") RAM peak over the suite | - | 52-54 GiB (watchdog: 54) | 41 GiB (32 GiB at the old 16 GiB pinned) |
| card 0's expert re-seed after a prompt | - | 1.0-1.8 s | 0.66 s; 0.17 s for prompts <= 4096 tokens (keep-cache) |

Hardware use now (`utilos.sh`, one-shot in the service's configuration; xpu-smi averages over the window):

| window | card 0 (dense half) | cards 1-3 (experts) |
| --- | --- | --- |
| 31.6K prompt (1210 tok/s) | 83 % busy (76 % before), reads 133 GB/s | 40 % busy each (31 % before, 17 % at goal start) |
| 8.5K prompt (778 tok/s) | 68 % busy, reads 99 GB/s | 46-48 % busy (42 % before) |
| DSpark decode, the essay (49 tok/s) | reads 170 GB/s (~30 % of the 580 GB/s peak) | reads ~56 GB/s each (33 before) |

What changed (commits 0514c542 .. 52468478, `larry-xe`, local):
- **xe_dqgemm** (`STRATA_XE_DQGEMM`): the prompt's expert GEMMs decode Q3_K / Q2_0 / ternary weights inside an XMX
  joint_matrix kernel (each lane decodes its own weight row straight into its B fragment; wide loads once a block;
  two values a 32-bit op as fp16) instead of writing fp16 copies for oneMKL: expert-card GEMM time 13.6 -> 8.3 s of an
  8.5K prompt, 1.7x the first form. Measured XMX fp16 peak on a B70: 170-177 TFLOPS (`bench/dqb/dpas_peak.cpp`).
- **Prompt buffers**: overflow experts kept for a layer's second column half (`STRATA_XE_KEEP_STREAM`), per-card call
  buffers sized once a prompt (the stream buffer regrew in 5 steps, 0.9 s).
- **F16 routers** (`STRATA_ROUTER_F16`: their values are bf16, exact in F16) and the **Engram KV projections** at Q6_K
  (`STRATA_ENGRAM_REQUANT`, the service sets q6_K) split over the cards: decode +4 % (they were 1.8 ms of a token).
- **Promotion cache** (`STRATA_XE_PROMOTE`): decode experts the CPU pool ran twice are copied into a slot cache on the
  expert cards (CPU repack into a pinned bounce, copied by a kernel on the card's own queue behind its experts):
  DSpark verify 45 -> 43 ms/pass, plain decode +3 %, VRAM hits 98.7 -> 99.5 %.
- **Native allocation** (`STRATA_XE_LOCAL_ALLOC`): SYCL device allocations go through UR / UMF, which make each one
  resident on every P2P peer; with larry's ACS-redirect card pairs that cost ~1 GiB of host RAM per GiB of VRAM
  (`bench/dqb/alloc_probe3.cpp`) - the expert slabs alone held ~14 GiB of RAM, card 0's re-seed another ~14 GiB and
  ~3 s. Slabs and flex regions now come from `zeMemAllocDevice` + `zeContextMakeMemoryResident` (own card only).
- **Keep-cache** (`STRATA_SERVE_KEEP_CACHE`): a prompt of <= 4096 tokens gives back only card 0's tail slabs it needs.
- `STRATA_XE_PINNED_GIB` 16 -> 24 (room freed by the above): 8.5K prompt +4 %.

Lessons (all measured; `bench/dqb/`): after a >= ~128 KB H2D `queue.memcpy`, the next kernel submitted to that idle
in-order queue starts ~300 us late (`defer_q.cpp`); `xe_repack_blob` takes 739 us for a Q3_K expert (byte-wise);
allocating ~1 GiB mid-decode stalls ~0.5 s; a second queue on a spinning card does not run; a copy from card 0's
copy engine into another card ran at ~11 MB/s and held up card 0's own copies.

What bounds it now: a long prompt is card 0's dense half (attention over the growing context); an 8K prompt streams
~55 GiB of RAM-arena experts once per batch (~15 GB/s: PCIe + host memory); decode is card 0's ~470 us of kernels a
layer (big mat-vecs at ~90 % of peak, ~50 small ones latency-bound) plus ~74 us waiting for the expert cards.

## Measured (2026-10-20, after the "double it" round)

One binary, two configurations, one-shot greedy runs (`told2.sh`): OLD = every switch added since the goal started turned
off plus the goal-start flags (`--pf-b 1.43`, D3 prediction, `--prompt-batch 8192`), NEW = today's defaults.

| one-shot (`run.sh`) | OLD | NEW |
| --- | --- | --- |
| decode, no DSpark (`p1.i32 -n 160`, 2 runs) | 16.8 tok/s | 36.6 tok/s (2.18x) |
| decode, no DSpark, the essay (`pe.i32 -n 400`) | 17.2 tok/s | 37.3 tok/s (2.17x) |
| DSpark decode (`p1.i32`) | 23.0 tok/s (verify 124 ms/pass) | 59.8 tok/s (48 ms/pass) (2.60x) |
| DSpark decode, the essay | 15.3 tok/s | 42.5 tok/s (2.78x) |
| prompt, 8.5K tokens (`p8k.i32`) | 321 tok/s (26.6 s) | 644-670 tok/s (12.7-13.3 s) (2.0-2.1x) |
| prompt, 31.6K tokens (`p30k.i32`, needle found) | 457 tok/s (69.3 s) | 926-988 tok/s (32-34 s) (2.0-2.2x) |

| service (public endpoint, server timings) | goal start (2026-10-19 evening) | now |
| --- | --- | --- |
| `svcbench.py` (4 short answers, DSpark) | 20.6 tok/s | 50.3-50.6 tok/s (2.4x) |
| `essay.py` (800 tokens, mean of 3) | 17.3 tok/s | 38.5-39.2 tok/s (2.2x) |
| prompt ~8K tokens (`needle.py 8000 1`: a cold prompt each time) | ~310 tok/s | 666-717 tok/s (the first long prompt after a start: 438-580) |
| prompt ~16K tokens | - | 918 tok/s |
| prompt ~31-32K tokens (needle found) | ~440 tok/s | 946-1114 tok/s (2.5x with the pass ring) |

Perplexity of the 8.5K prompt text (`--ppl`) stays 1.676-1.680 on every path (f32 or fp16 prompt GEMMs, serial or
pipelined); greedy text is not run-to-run deterministic (pre-existing: the layer-major prompt's last digits vary and the
indexer's top-k amplifies them), so compare logits / ppl, not bytes.

Hardware use (`utilos.sh NAME old|new IDS N`: xpu-smi on all four cards, averaged over the prompt or decode window cut
from the engine's log; one-shot, same binary, OLD = the goal-start switches):

| window | card 0 (dense half) | cards 1-3 (experts) |
| --- | --- | --- |
| 31.6K prompt, OLD (459 tok/s, 69 s) | 77 % busy, reads 85 GB/s, 195 W | 17 % busy, 10 GB/s, 87 W |
| 31.6K prompt, NEW (964-1082 tok/s, 29-33 s) | 76 % busy, reads 108 GB/s, 183 W | 31 % busy, 24 GB/s, 102 W |
| 8.5K prompt, NEW (637 tok/s) | 66 % busy, 80 GB/s | 42 % busy, 27 GB/s |
| decode, essay, OLD (17.1 tok/s) | reads 125 GB/s | 13-14 GB/s |
| decode, essay, NEW (36.5 tok/s) | reads 249 GB/s (~43 % of the ~580 GB/s peak) | 33-34 GB/s |

(xpu-smi's "compute busy" reads ~100 % for every card during decode - the cards spin-wait - so decode is shown by
memory reads.) What still bounds it: a long prompt is card 0's dense half (attention over a growing context: ~20 s of
a 31.6K prompt); an 8K prompt is the expert tier (it dequantises each expert once per column half, and ~1,300 arena
experts a prompt do not fit the cards' prefetch buffers - the expert cards end a call with 0.5-0.6 GiB free); decode
is card 0 streaming its 6.8 GB a token plus ~45 small kernels a layer and the cards' ~100 us round trip - a batch-1
MoE touches 6 of 384 experts a token, so the expert cards' bandwidth mostly holds capacity.

Per decode layer against the hardware (unitrace kernel times, 42 tokens; one token = 40 layers, ~685 us each at 36.5 tok/s):

| part of a decode layer on card 0 | time | vs the hardware |
| --- | --- | --- |
| q_b, wo_b, shared gate+up mat-vecs (Q6_K R8, ~29 MB each) | ~170 us | ~520 GB/s = ~90 % of the measured 580 GB/s streaming peak |
| wo_a (8 per-group slices, 27.5 MB) | ~67 us | ~410 GB/s (70 %) |
| the small projections (q_a, kv, compressor, indexer, hc mixes: ~25 MB) | ~110 us | latency-bound kernels of 2-27 us |
| ~35 other kernels (norms, quant_x, router, hc pre/post/comb, rope, gathers, attention ~25 us) | ~110 us | 1-8 us each |
| waiting for the expert cards' sum | ~95 us | host dispatch 32-69 us, cards ~75 us, card-to-card return ~50 us |
| gaps between dependent kernels | ~100 us | ~1-3 us per launch |

So card 0's floor at this weight size is 170 MB / 580 GB/s = ~293 us a layer (~85 tok/s) if every kernel streamed at peak
with no latency; the big mat-vecs are already at ~90 % of it. What is left is latency, not bandwidth: fusing the ~35
small kernels into a few and dispatching the expert cards from the GPU (no host round trip) would be the next ~20-25 %,
each a multi-day kernel project. The expert cards cannot be kept busy by one stream: a token reads 6 experts a layer,
~21 MB a card, ~37 us at peak against a ~685 us layer (<= ~6 % duty) - only more tokens a pass (DSpark's 2.5-3.3,
or batching several conversations) raises that. Tried this round and kept off (no gain): card 0 taking the prefetch
overflow (`STRATA_XE_PREFETCH_DENSE`), overlapping the overflow stream with compute (`STRATA_XE_OVERLAP`), no lock
priority for the dense half (`STRATA_SUBMIT_PRIO=0`); unitrace cannot trace the pipelined long prompt (it crashed /
timed out twice; untraced runs under glibc's heap checker were clean).

Where an 8.5K prompt's ~13 s goes now: the dense half (card 0) ~8-9 s of the pipelined batch, the expert tier's exposed
part ~2-3 s (its ~10 s run under the dense half: each layer in two column halves, the tier on one while card 0 runs the
other), ~1 s outside the layer loop. Before the round: dense 15 s then experts 11-13 s, one after the other - card 0 lost
~5.7 s gathering attention keys through get_rows / concat / cast and ~3 s in f32 GEMMs (no XMX), the expert cards waited
for 69 GB of arena experts streamed on their critical path, and each side idled while the other worked.

DSpark stays at depth 4 / confidence 0.6: a 2026-10-20 re-sweep put depth 5 (conf 0.5 or 0.6) within noise of it (essay 35.8 vs 36.0-37.1, svcbench 41-46 for both).

Where a decode layer's time goes now (host-clock stamps, `STRATA_GPU_TL=1`): card 0 runs attention ~485 us (its Q6_K
mat-vecs stream ~146 MB at ~440 GB/s, plus ~45 small kernels), the shared expert ~85 us, and waits ~95 us more for the
expert cards' sum (each card ~75 us from dispatch, the host dispatching them one after another); a verify pass (5.5 tokens) ~780 us of
attention and ~300 us of the expert tier per layer, of which the CPU pool's arena experts (~0.5 a layer at ~480 us) are
the largest part. xpu-smi over the essay (`utilprobe.sh`; averages over its window, idle included): card 0 reads
~154 GB/s (it was ~94 GB/s before the round), cards 1-3 ~20 GB/s each - card 0's dense half stays the limit; the expert
cards are mostly capacity.

## Measured at first light (2026-10-19, through the public endpoint unless noted)

| | |
| --- | --- |
| load (`larry-mode ds41` to answering) | 80-95 s (experts from the page cache; the first start after a reboot reads 160 GiB from NVMe: ~3 min) |
| decode, short answers (DSpark speculative decoding on) | 20-28 tok/s (code ~24-28, prose ~19-22) |
| decode, one-shot without DSpark (`run.sh`) | 15-16 tok/s |
| decode after an 8.5K-token prompt | ~18.7 tok/s |
| prompt, 8.5K tokens (cold) | ~310 tok/s (27 s: dense half 15 s, experts 12 s) |
| prompt, 31.7K tokens (cold; needle in the middle found) | 410-440 tok/s (72-77 s) |
| short prompt (20-60 tokens) | 0.4-2 s; the first request after a load also loads the DSpark draft (~5-15 s) |
| expert hits from VRAM while decoding | 97-99 % (route bias x0.5 toward resident experts, Xolotl's coding-safe default) |

Quality checks: answers verified by hand (code, arithmetic with thinking, a long-document summary, an OpenAI tool-call
round trip, a passphrase needle at 31.7K tokens); the Xe expert kernels
match a dequantised F32 reference to ~2e-7 relative (`test_ds41_xe`), better than the CPU tier's 8-bit activations (~1e-2).

## How it runs

| Piece | Where |
| --- | --- |
| dense half (attention, router, shared expert, Engram add, head; Q8_0 requantised to Q6_K at load) | card 0 (`xpu0`, 0000:15:00.0), ggml's SYCL backend |
| routed experts, hottest first | cards 1-3: ~8,700 experts / 84 GiB, permanent; card 0: ~1,000 more / 10 GiB, given back during long prompts |
| the other ~6,600 experts | RAM arena, 68 GiB: 16 GiB pinned (SYCL host USM), the rest ordinary memory |
| Engram tables (2 x 95 GiB, ~48 small random reads a token) | one per NVMe: `models/gsq1017/engram-layers.1` (/opt), `/srv/dsv41/gsq1017/engram-layers.14` (/) |
| DSpark (the checkpoint's own 3-layer MTP draft stack, Q2_K sidecar, 5.1 GB) | card 0, loaded with the first request |

Per decode step and layer, the dense half runs on card 0, the router picks 6 experts, and every card computes the ones it
holds (one gate/up+SwiGLU kernel and one down+weighted-sum kernel per card) while the CPU pool computes any arena expert;
the sums land in the dense half's tensor on card 0. A prompt runs expert-major per 8192-token batch: experts with >= 8
tokens as two fp16 oneMKL GEMMs (XMX) after a one-time dequantisation, the rest as mat-vec jobs, arena experts streamed
onto the least-loaded card, one host thread per card, the cards' parts combined card-to-card on card 0.

Decode / verify since 2026-10-20 is asynchronous on card 0: the kept graphs (attention, shared expert, finish) are queued
with `ggml_backend_graph_compute_async`; the host waits only for the router's readback (an event after three D2H copies
into **pinned** mirrors - pageable ones made every "async" copy block), queues the shared expert before that wait,
dispatches the cards (their X read straight from the pinned mirror), runs the CPU pool's experts on a helper thread meanwhile,
and queues the routed sum and the next layer without waiting. The pass inputs go up from a pinned staging buffer.

## Files

| Path | What |
| --- | --- |
| `xolotl/` | Xolotl checkout, branch **`larry-xe`** (my commits on top of upstream `f7c561b7`) |
| `xolotl/tools/v41/engine/xe/` | **the Intel tier**: `ds41_xe_tier.cpp` (placement, decode/verify, prompt path), `xe_kernels.cpp` (SYCL kernels for Q3_K / Q2_0 / PTQ1_0 experts, fp16 dequant), `xe_cpus.hpp` (host thread placement), `test_ds41_xe.cpp`, `test_xe_q2k.cpp` (the Q2_K form + the repack forms), `bench_cpu_pool.cpp` (the CPU pool against a streaming read), `xe.cmake`, `ggml-sycl-dsv41.patch` (Xolotl's fused DSV4.1 ops, XMX prompt attention and indexer for ggml-sycl) |
| `xolotl/tools/v41/engine/tp/` | **engine1** (`STRATA_TP=N`): `tp_core.cpp` (the N-card decode / verify core), `tp_kern.hpp` (its kernels), `tp_pager.cpp` (card 0's copies of the sharded weights, paged: `STRATA_TP_PAGE`), `tp_bench.cpp`, `tp.cmake`, `DESIGN.md`; `bench/engine1/` its scripts and probes |
| `xolotl/build-xe/` | the build: `ds41_generate_xe`, `libds41_xe.so`, `test_ds41_xe` |
| `models/gsq1017/` | the GGUF (`V41-GSQ-1017.gguf`, sha256 014ef21a...), `rank.bin`, tokenizer, DSpark sidecar, Engram L1 |
| `/srv/dsv41/engram-epg/engram-L1.epg` | layer 1's Engram table as 4 KiB pages of 15 rows (one read a row; on the WD SN850X), written / verified by `bench/engram/engram_epg.py`; first in `--engram` (the repo's shards untouched). `bench/engram/iops.c` (random-read rates of the drives), `test_engpg.cpp` (the reader's rows against the shards) |
| `larry-ds41.json` | server config: engine flags (`--ctx 262144 --route-bias 0.5 --route-bias-file 0.3 --prompt-batch 16384 ...`), sampling |
| `ds41-xe-engine.sh` | the engine as the server starts it: oneAPI env, DSpark/serve settings, the tier's knobs |
| `ds41-serve.sh` | the unit's launcher: API key, then `serve_ds41.py` on 172.23.0.1:18080 |
| `/etc/systemd/system/larry-ds41{,-watch,-recover}.service` | the service, its memory watchdog (driver RAM 54 GiB / 6 GiB floor), the recovery to `small` |
| `logs/engine.log`, `logs/watch.log`, `journalctl -u larry-ds41` | engine log, watchdog log, server log |
| `build-xe.sh`, `bootstrap.sh` | rebuild; set everything up from nothing (clone, patches, build) |
| `run.sh NAME [args]`, `tok.py enc/dec`, `xe-env.sh`, `vt.sh` | one-shot runs (logs/NAME.*), prompt <-> ids, the environment, VTune |
| `dev-hold.sh on` | free all four cards for development without a layout (`larry-mode small` undoes it) |
| `bench/speed/` | the standard speed table: `run_all.sh` (the service held, a DSpark and a plain private copy, the service back), `tspeed.sh NAME [VAR=..]` (one copy: warm-up, `vbench.sh` = `vllm bench serve` on MT-Bench, `speedtab.py` = llama-bench's pp512 / tg128 at depths), `patch_vllm_chat.py`, `hf/` (tokenizer + MT-Bench cache), results in `out/`; the prompt A/B: `tab_all.sh NAME:VAR=v,.. ...` (the service held, `tab.sh` a config: speedtab's `TAB_SPECS`, then the service back; `TAB_PPL=1` adds `tlppl.sh` = one-shot prompt-path logits KL via `lkl.py`), `tabsum.py` (the table), `tpg_light.sh`, `ab*.sh` (the night's A/B windows, each after the AIME client's current problem), `run_all2.sh` + `sttab.py` (the speed table again; old vs new as markdown); the afternoon A/B window `abw.sh NAME:TESTS[:VAR=v,..] ..` (waits until the service has been idle `AB_IDLE_MIN` minutes, `yieldsrv.py` answers 503 on :18080 and ends the window if a request comes; `tabx.sh` one configuration, `abx.py` prompts from empty / the edit / decode tasks, `abxsum.py` the table), `turns.py ENGINE_LOG [LINE]` (per request: prompt, decode, ms and CPU-pool experts a pass); the evening's `agentx.py` (an agent session replayed: the test "agent", `AGENT_ARGS`), `abx.py --needles N,..` (the test "needles", `AB_NEEDLES`), `AB_CTX` (the config's `--ctx`), a window step `@COMMAND` (a probe or tpg.sh run as it is); `bench/dqb/stage_ring.cpp` (pageable -> cards: the tier's stage vs rings in the L3) |
| `bench/li/` | the prompt indexer: `li_probe.cpp` (its kernels one by one), `topk2_test.cpp` + `topk2.sh` (the two-stage top-k against the radix select: sets and times), `lifbf_probe.cpp` + `lif_kernel.inc` + `lifbf.sh` (the fused kernel's fp16 copies against bf16 in place) |
| `bench/longctx/` | long contexts: `xbuild.sh NAME PATCH..` (an engine change built at low priority and staged in `NAMEbin`, the service's build put back), `xinstall.sh NAME` (staged -> `build-xe`, an abw step), `r1m-ra.sh` (131K + 984K needles on a staged engine with the replica check), `ra-ab.sh` (the read-ahead off / on, with vmwatch), `prof1m.sh` + `profdiff.py` (card 0's op profile over a long prompt, by depth), `tsvc2-ra.sh` (tsvc2 + concurtest after an install), `pp-ab.sh` (one long needle a configuration), `lt1m.sh` (`r1m-ra.sh` a configuration) |
| `bench/quality/` | AIME 2026 through the service at DeepSeek's settings (DeepInfer `eval/aime`: its prompt and rule judge, effort 100, top-p 1.0, 144K, one sample a problem): `aime.py [--deadline HH:MM] [--score]` (resumable; a `STOP` file in `out/` ends it after the current problem), `data/aime_2026.jsonl` (MathArena/aime_2026), `out/aime-1.jsonl` + `.log`, `encchk.py` / `effchk.py` (the prompt matches DeepSeek's encoder; the effort reaches the engine); `gpqa.py` + `mkgpqa.py` + `data/gpqa_diamond.jsonl` (GPQA Diamond the same way: DeepInfer's prompt and rule judge, the public mirror hendrydong/gpqa_diamond_mc; not run yet). Both clients YIELD: when the server's /metrics shows another request queued they cancel their own and stop (`fakesrv.py` tests that) |

## Knobs (environment of `ds41-xe-engine.sh`; `larry-mode restart` after a change)

| Variable | Default | |
| --- | --- | --- |
| `STRATA_XE_PINNED_GIB` | 24 (was 16) | pinned part of the RAM arena (all GPU-driver host memory is capped by TTM at ~60.7 GiB on larry) |
| `STRATA_XE_DQGEMM` | 1 | prompt: the dequantising XMX expert GEMM (0 = fp16 copies + oneMKL) |
| `STRATA_XE_KEEP_STREAM` | 1 | prompt: overflow experts stay on the card for the layer's second column half |
| `STRATA_XE_PROMOTE` (`_MIN` 2, `_CHUNK_MB` 1, 4 under TP) | 1 | decode: CPU-pool experts promoted into a slot cache on the expert cards (`_CHECK=1` verifies each slot) |
| `STRATA_XE_FLEX_GIB` | 1.0 | per expert card, allocated at load: the prompt's prefetch buffers / decode's promotion slots |
| `STRATA_XE_LOCAL_ALLOC` | 1 | expert slabs and flex regions via `zeMemAllocDevice` (0 = SYCL USM: ~1 GiB host RAM per GiB here) |
| `STRATA_SERVE_KEEP_CACHE` (`STRATA_SERVE_KEEP_MAX` 32768 in the service, engine 4096) | 1 | serve: a batch of at most KEEP_MAX new tokens gives back only card 0's tail slabs it needs (sized with `STRATA_SERVE_BATCH_FIXED_GIB` / `_COL_MIB`; no ~1 s re-seed after it) |
| `STRATA_PIPE_TRACE`, `STRATA_XE_PF_TRACE` | 0 | diagnostics: the pipelined prompt batch's steps per layer (dense parts, finishes, tier joins / launches); each prefetched layer's plan / staged / DMA-done times and every chunk call's wall split |
| `STRATA_ADDR_LOG` | 0 | diagnostic: every device allocation / free of the tier, every virtual-memory reserve / map / unmap of the TP core and the pager (with `GGML_SYCL_MEMTRACE=2`: a GPU fault's address matched to its buffer) |
| `STRATA_ROUTER_F16` | 1 | the F32 routers stored F16 (bf16 values: exact) |
| `STRATA_ENGRAM_REQUANT` | q6_K (service) | the two Engram KV projections re-quantized (engine default: off) |
| `STRATA_HEAD_SPLIT_MB` | 100 | Q6_K mat-vecs from this size are split over the cards (the LM head, the Engram KV) |
| `STRATA_XE_RESERVE_GIB` | 3.5 | VRAM per expert card kept free for the prompt path's buffers |
| `STRATA_XE_STREAM_FIT` (`_MARGIN_GIB`) | 1 (0.5) | a prompt call's stream buffer grows only into its card's free VRAM beyond the margin and 100 KiB a token (more rounds instead; the 2026-10-26 03:16 abort); 0 = up to the planned size |
| `STRATA_XE_DENSE_GIB` / `STRATA_XE_DENSE_RESERVE_GIB` | 10 / 8 (service 14 / 1.75, was 4 until 2026-10-24 22:25) | experts on card 0, and the VRAM left there for ggml's pools |
| `STRATA_XE_GEMM_MIN` | 8 | prompt: experts with at least this many tokens take the GEMM path |
| `STRATA_DENSE_ASYNC` | 1 | 0: ggml's synchronising graph compute per decode graph, pageable-style waits (the old path) |
| `STRATA_XE_XDIRECT` | 1 | 0: stage X through each card's staging buffer instead of reading the pinned mirror |
| `STRATA_XE_CPU_ASYNC` | 1 | 0: the arena experts after the cards' dispatch on the main thread |
| `GGML_SYCL_FASTMV_R8M` | 1 | 0: the generic Q6K_R8 mat-vec (1: row-blocked, 256-register mode for >= 16 MB matrices) |
| `STRATA_PREREORDER` | 1 | the dense Q6_K weights' planar reorder at load (0.15 s), not in the first decode token (~0.4 s) |
| `STRATA_HEAD_SPLIT` | 1 | the LM head (and DSpark's shared head) split by rows over all four cards (card-to-card copies) |
| `STRATA_SHEXP_GU` | 1 | the shared expert's gate + up as one weight over their adjacent bytes |
| `STRATA_ENG_ASYNC`, `STRATA_RAW_F32`, `STRATA_XE_CPU_ASYNC`, `STRATA_XE_XDIRECT` | 1 | Engram reads under layer 0; window ring in F32; CPU-pool experts on a helper thread; cards read X from the pinned mirror |
| `GGML_SYCL_LARRY_QREUSE`, `_CPYBATCH`, `_NORM_SMALL`, `GGML_SYCL_FASTMV_R8M` | 1 | ggml-sycl: one quantisation per shared input; batched copies; short-row norms on a work-group; row-blocked / 256-register mat-vecs |
| `STRATA_XE_HSLOTS`, `STRATA_XE_LPR=16`, `GGML_SYCL_LARRY_NORM_WG`, `GGML_SYCL_DSV4_DEC_XMX=0` | off | measured slower (host-memory expert parts, 16 lanes a row, smaller norm groups, FMA decode attention) |
| `STRATA_XE_GRAPH_MULTI`, `GGML_SYCL_GRAPH_CACHE` | 0 | replayed SYCL graphs for verify calls / card 0's graphs: no gain measured (the latter also changes tokens) |
| `STRATA_DEC_PROF`, `STRATA_XE_PROF`, `STRATA_GPU_TL`, `GGML_SYCL_GRAPH_STATS` | 0 | diagnostics: host split per layer, tier phases, device timeline (perturbs), graph-cache counters |
| `STRATA_KGATHER` | 1 | prompt attention: each query's window + top-k compressed keys gathered straight into f16 (`GGML_OP_DSV41_KGATHER`); 0 = get_rows / concat / cast |
| `GGML_SYCL_LARRY_F16GEMM` | 1 | prompt-sized (>= 32 columns) mat-muls of quantised / F16 weights on the fp16 XMX path (f32 accumulation); 0 = f32 GEMMs |
| `STRATA_PIPE` | 1 | the layer-major prompt batch in two column halves per layer, the tier's half overlapping card 0's other half |
| `STRATA_XE_PREFETCH` (`_MIN` 2048 engine / 4096 service, `_MIN_B` = `_MIN` engine / 2048 service: an enc-skip batch's upper layers, `_STAGE_GIB` 1.5, `_MARGIN_GIB` 0.5, `_STREAM_GIB` 0.25), `STRATA_XE_STAGERS` 4 | 1 | prompt: layer l+1's arena experts staged (4 threads) and copied onto the expert cards while layer l runs |
| `STRATA_XE_XP2P` | 1 | pipelined batch: the expert cards copy X card-to-card from card 0 (no host round trip) |
| `STRATA_PASS_RING` | 1 | pipelined batch: card 0's span graphs alternate two allocators (event-guarded) instead of a device wait per span |
| `NEOReadDebugKeys=1 ExperimentalH2DCpuCopyThreshold=0 ExperimentalD2HCpuCopyThreshold=0` | set in `xe-env.sh` | the driver's CPU path for small copies waits for the queue to go idle (~5 ms a span with the ring); decode measured the same |
| `STRATA_XE_PREFETCH_DENSE` | 0 | 1: card 0 takes the prefetch overflow (no gain measured) |
| `STRATA_XE_OVERLAP` | 1 (service; engine 0) | prompt calls: a card's overflow experts DMA'd on a second queue while it computes its resident / prefetched ones first (pageable ones through its hot ring) |
| `STRATA_SUBMIT_PRIO` | 1 | 0: the dense half gets no priority on the submit lock (8.5K ~5 % slower) |
| `--prompt-batch` | 16384 | one pipelined batch up to 16K tokens (each batch streams every arena expert once) |
| `STRATA_SERVE_BATCH_FIXED_GIB` / `_COL_MIB` | 3.5 / 0.45 (engine default 4.3 / 0.95) | the serve batch-width fit: card 0's need = fixed + per column (Xe: ~0.32 MiB of state a column there) |
| `STRATA_SERVE_DSPARK_EVICT` | 0 | >0: free DSpark's 5 GB for batches of >= N tokens (reload ~3-4 s); with the fit above a 16K batch fits beside it |
| `STRATA_SERVE_DSPARK` | the sidecar | empty = no speculative decoding |
| `STRATA_DENSE_REQUANT` | q6_K | empty = keep the dense Q8_0 (slower decode, 1.7 GB more VRAM) |
| `STRATA_PREREORDER_SLICES` | 1 | wo_a reordered per group slice at load: every prompt span takes the per-slice GEMM |
| `GGML_SYCL_LARRY_F16CACHE_MB` / `_FREE_MB` | 768 / 3072 | prompt GEMMs keep fp16 copies of their weights across a batch's spans (0 = off) / VRAM kept free |
| `STRATA_HC_FUSE_MAXN` | 16384 (service, since 2026-10-26 20:10; engine 8) | the fused hyper-connection ops for passes of at most this many tokens - prompt spans too since patch-hcwide.py's wide hc_fa (abwT144: fresh 16K / 31K -3-4 %, 984K 357 -> 346 s); 8 = decode widths only |
| `STRATA_ENG_PIN`, `STRATA_ENG_PREFETCH` | 1 | prompt Engram rows in pinned memory + async upload; the next batch's rows read ahead (`STRATA_ENG_PROF=1` prints the phases) |
| `GGML_SYCL_LARRY_MMID_Q2K`, `GGML_SYCL_LARRY_ARGMAX` | 1 | DSpark draft: device-side Q2_K experts for 2-8 tokens; argmax of long rows over 64 work-groups (`_CHECK=1` compare) |
| `STRATA_DSPARK_DENSE_TYPE` | q6_K (service) | the draft's Q8_0 projections requantized on load (engine default: off); `STRATA_DSPARK_PROF=1` the draft's host phases |
| `STRATA_ENGRAM_THREADS` | 256 (service) | the Engram row reader's pool = its queue depth over both NVMe drives (engine default 32) |
| `STRATA_TP_OPROJ` | 0 | 1 / 2: the attention output projection split over the four cards (in-graph op / the attention graph in two halves); needs `STRATA_XE_RESERVE_GIB` 4.2; `STRATA_TP_GRAPH` (1) records each card's part, `STRATA_TP_PROF=1` its host cost |
| `STRATA_XE_XMX` | 0 | 1: decode / verify experts on XMX (flat in tokens, slower below ~4 tokens); `STRATA_XMX_KSG` / `_KSD` / `_GRF` its K splits / register mode |
| `GGML_SYCL_LARRY_XMXMV` | 0 | 1: XMX multi-column Q6_K mat-vecs for verify passes (no end-to-end gain measured) |
| `GGML_SYCL_LI_GEMM_MIN` / `GGML_SYCL_LI_XMX` | 1 / 1 (service; engine 32 / 0) | the lightning indexer of decode / verify passes on XMX: li_gemm for every batch / one fused bf16 kernel for <= 8 queries (`GGML_SYCL_LI_XMX_CHECK=1` compares with li_gemm) |
| `GGML_SYCL_LI_FUSED` (`_MIN`, `_QW`, `_KR`, `_CHECK`, `_CHECK16`) | 1 (service, since 2026-10-26 07:48; engine 0) (16, -, -, 0) | the prompt indexer (batches of >= `_MIN` queries) as one fused XMX kernel - the score matrix stays in registers (16K keys 4.36 -> 0.63 ms); `_QW` / `_KR` override the work-group's queries / keys, `_CHECK=N` compares the first N calls with li_gemm; 2 = the BF16 keys read in place (the same bits; 6 % slower at 512K keys: not used) |
| `GGML_SYCL_TOPK2` | 1 (service since 2026-10-28 03:12; engine 0) | ggml-sycl's top-k of long rows (>= 32K columns, >= 4k blocks) in two stages: the blocks' maxima, then the radix select over the blocks that reach the k-th of them - the same set, 2.6x at 512K columns |
| `STRATA_LI_TOPK` (`GGML_SYCL_LI_TOPK_MIB` 512, `_CHECK` 0, `_CHECK_KEYS` 131072) | 1 (service since 2026-10-28 05:06; engine 0) | the indexer of every layer that ranks all visible rows (2 / 8 / 14) as one op with its top-k (`ggml_lightning_indexer_topk`: no [cap, n] scores in the graph); `_CHECK=N` checks N calls' ids against card 0's scores |
| `STRATA_LI_REMOTE` (`GGML_SYCL_LI_REMOTE_SHARE` 25 engine / 35 service, `_MIN_KEYS` 65536, `_PRIO` 0) | 0 (service since 10-28 17:59, was 131072; engine 0 = off) | a prompt of >= this many tokens hands that op's queries to cards 1-3 (card 0 keeps `_SHARE` % of each call of >= `_MIN_KEYS` keys; their fp16 key copies, 657 MiB a card at 1M, freed after the prompt); `_PRIO=1`: their queues at low priority (measured worse) |
| `STRATA_LI_NVIS` | 1 (service, since 2026-10-26 07:48; engine 0) | the prompt path's indexer takes each query's visible row count (`ggml_lightning_indexer_nvis`) instead of a [cap, n] 0 / -inf mask built per call; ggml-sycl only |
| `STRATA_OPROJ_A` | 1 (service, since 2026-10-26 10:23; engine 0) | the prompt path's wo_a as one op (`ggml_dsv41_oproj_a`): its 8 slices in one oneMKL gemm_batch written in wo_b's input layout (768 us + a 162 us copy -> 520 us a 512-column span) |
| `STRATA_HC_NORM_MV_WIDE` / `GGML_SYCL_NORM_MV_WIDE` | 1 (service, since 2026-10-26 10:23; engine 0) / 1 | the prompt path's hc mixes through `ggml_dsv41_norm_mv` / ggml-sycl's wide kernel for it (> 8 tokens; 0 = the per-row one) |
| `GGML_SYCL_LARRY_SLICES_BATCH` | 0 | `larry_slices_gemm` (wo_a through mul_mat) as one strided gemm_batch into the [OL, n, OG] layout (`STRATA_OPROJ_A` supersedes it) |
| `GGML_SYCL_OP_PROF_SHAPES` | 0 | with `GGML_SYCL_OP_PROF=1`: each label also carries the node's shape and its first source's name / type |
| `STRATA_SERVE_KEEP_LEARN_DEPTH_GIB` | 2.0 (service, since 2026-10-26 19:45; engine 0) | a kept-cache prompt batch gets this many GiB * min(1, conversation tokens / 1M) more room on card 0 (shrink and width): a 984K conversation's 11.7K turn needed 10.54 GiB, more than either estimate (abwT141-143) |
| `STRATA_XE_LOAD_DIRECT` | 1 | the tier's bulk reads at a start (the experts placed in VRAM and the arena, the replicas) through the blob source's O_DIRECT twin: 156-198 GB -> 2-6 GB through the page cache a start (abwT139); 0: the buffered reads |
| `STRATA_ENGRAM_DROP` | 1 (service, since 2026-10-26 18:50; engine 0) | a prompt batch drops the Engram weight pages it read from the page cache (DONTNEED): no TTM backups / swap-ins from Engram's 12-14 GB a prompt set (abwT140); a repeated big prompt ~4 % slower |
| `STRATA_DSPARK_CUM` | 0.5 (service, since 2026-10-26 17:30; engine 0 = off) | DSpark: the verify window also stops where the product of the drafts' confidences so far drops below this (besides `STRATA_DSPARK_CONF`'s floor on each; Project Maya's rule): 102.7 -> 106.2 tok/s on the decode tasks (abwT137) |
| `STRATA_XE_JOBNT` | 1 (service; engine 0) | each decode / verify job's own token count picks its expert kernel form (2 = separate gate/up launches per width: slower) |
| `STRATA_XE_COMBINE` | 2 (service; engine 0) | prompt chunks: each expert card pushes its part to the dense card when it is done (2: as fp16, half the bytes) |
| `STRATA_XE_FN16` (`_MIN` 0) | 2 (service since 10-28 17:59; engine 0) | prompt calls: the call's X converted to fp16 once on the dense card; 1: the expert cards copy it (copy engine), 2: they read it by peer loads - half the bytes over the dense card's link |
| `STRATA_XE_DQ2` | 1 (service since 10-28 18:57; engine 0) | the prompt path's expert GEMMs, xe_dqgemm v2 (patch-dqv2.py): each chunk's weights decoded once into local memory VNNI-packed, a sub-group 32 tokens x 64 rows - the same bits, 1.4-1.8x |
| `STRATA_XE_CPU_LIGHT` | 3 (service since 2026-10-25 22:14, was 4; engine 0) | prompt calls: an arena expert the call would stream that serves <= T tokens (at most 8) runs on the CPU pool beside the cards; `_PINNED=0` streams pinned ones as before, `_MAX_N` (4096) bounds the calls |
| `STRATA_XE_PREFETCH_DEPTH` | 5000 (service; engine 0) | prompt: a batch that ends before this context position takes no prefetch (its arena experts stream on demand, light ones on the CPU pool; the promotion cache stays) |
| `STRATA_XE_DENSE_LITE` (`_MIN` 0) | 1 (service; engine 0) | pipelined prompt calls: the dense card reads X in place, queues its part without host waits, and (in calls of >= _MIN tokens) takes no streamed experts |
| `STRATA_XE_DENSE_INLINE` | 1 (service; engine 0) | pipelined prompt calls: the dense card's part and the combine on the dense half's own queue |
| `STRATA_XE_DENSE_OFFLOAD` | 4096 (service; engine 0) | pipelined prompt calls of >= N tokens: the dense card's cached experts computed by the expert cards from their pinned arena copies |
| `STRATA_XE_GATHER_INIT` | 1 (service; engine 0) | prompt calls: a card's first gather writes its part (no memset) |
| `STRATA_XE_SEED_TAIL_LAYERS` | 20 (service; engine 0) | card 0's cache laid out with layers >= L first (a batch's keep-cache shrink gives back the lower layers' experts) |
| `STRATA_EMB_PIN` | 1 (service; engine 0) | prompt batches: the token table pinned at load, a batch's rows staged f16 and uploaded async (i_emb f16) |
| `STRATA_XE_PIN_STREAM` | 1 (service; engine 0) | the pinned arena for what prompts stream (card 0's cache below `STRATA_XE_SEED_TAIL_LAYERS`, then the next-ranked experts above it) |
| `STRATA_XE_ZCOPY` | 1 (service; engine 0) | a tier call's small uploads by kernels from pinned per-call copies (no copy-engine FIFO, no host waits) |
| `STRATA_XE_LT_COMPACT` | 1 | a call's CPU-light experts as compact rows: their X rows gathered on card 0, their sum scattered in after the combine (0 = the full-width form) |
| `STRATA_XE_PF_NOBLOCK` | 0 | a layer's last call does not wait for the next layer's staging; the slot's barriers recorded by its last stager (no gain measured) |
| `STRATA_GPU_PTL` | 0 | diagnostic: a prompt batch's GPU timeline to logs/ptl.PID.tsv (`bench/speed/ptlsum.py`, `ptlcall.py`) |
| `STRATA_STALL_WATCH` | 1 | diagnostic: a batch head running past 300 ms has its thread's kernel stack sampled and printed |
| `STRATA_SERVE_KEEP_LEARN` (+ `_COL_MIB` 0.6, `_MARGIN_GIB` 1.5) | 1 (service; engine 0) | the keep-cache shrink sized from the last kept batch's measured VRAM low point (turns no deeper than it or 16K) |
| `STRATA_SERVE_KEEP_LEARN_HIST` | 1 (service; engine 0) | the learned size from the last 8 kept batches: the smallest VRAM a batch as wide took, x the depth ratio for a deeper turn |
| `STRATA_SERVE_REGROW_ASYNC` | 1 (service; engine 0) | the regrow after a kept batch left running; published before the first TP pass after its copies (TP decode only) |
| `STRATA_SHEXP_TIER` (+ `_MIN` columns) | 1, MIN 4096 (service; engine 0, MIN 256) | a pipelined prompt part's shared expert on the expert cards instead of card 0 |
| `STRATA_XE_CPU_LIGHT_PINNED` | 0 (service; engine 1) | the CPU pool takes pinned arena light experts too (0: they stream by DMA) |
| `STRATA_PIPE_MIN_PART` | 0 | a pipelined batch's layer splits only when both parts have this many columns (kept whole: one part on the pipelined path) - measured slower for 513-767-token batches |
| `STRATA_XE_BAR_STAGE` | 1 (service; engine 0) | the expert cards' flex regions / stream buffers written by the CPU through their BARs (xe uAPI buffer objects imported into Level Zero): pageable experts without rings or DMAs |
| `STRATA_XE_STAGERS` | 4 (1 with `BAR_STAGE`) | the prompt prefetch's stager threads |
| `STRATA_XE_HEAD_STAGE` | 1 | the batch head's LM head split copies through tier-owned card-0 staging (0: the expert cards copy from / into ggml's tensors - dma-buf exports, the slow heads) |
| `STRATA_XE_HEAD_STAGE_CHECK` | 0 | diagnostic: the first 4 staged heads of each width also the direct way, compared ("head stage check") |
| `STRATA_XE_ALLOC_LOG` | 0 | diagnostic: every tier buffer growth on stderr (always: those over 30 ms) |
| `STRATA_XE_LT_PROF` | 0 | diagnostic: a prompt call's CPU-light part split (setup, read back, jobs, pool, sum), every 100 calls |
| `STRATA_STALL_PERF` / `STRATA_STALL_GDB` | 0 | diagnostic: a stalled batch head gets a perf call graph (300 ms) / every thread's gdb backtrace (400 ms) in logs/stall.PID.N.* |
| `STRATA_XE_LT_QUANT_THREADS` | 0 | the CPU-light activations quantised on N helper threads (no gain measured) |
| `STRATA_STALL_PERF` | 0 | diagnostic: a stalled batch head's thread perf-sampled into logs/stall.PID.N.perf |
| `STRATA_ATTN_SDPA` | 1 (service; engine 0) | the prompt attention of a full 512-query span (layers 0-19: 128 / 640 keys) as one oneDNN graph SDPA, the sink as an extra key; `STRATA_ATTN_SDPA_DP` the key width (576) |
| `STRATA_ATTN_FSA` (`GGML_SYCL_FSA_SLOTS` 64) | 1 (service; engine 0) | with `STRATA_ATTN_SDPA` and `STRATA_SDPA_O16`: those spans' attention as one kernel that reads each query's window rows (an fp16 copy of the span's window by position) and selected compressed rows where they lie - no gathered [n, 656, 576] copy (patch-fsa.py); `_SLOTS` the persistent work-groups |
| `STRATA_DUMP_SEL` | unset | debug (patch-dumpsel.py, not in the engine): every prompt batch's index-source selections appended to logs/dump_sel.PID.bin |
| `STRATA_XE_HOT_PROF` | 0 | diagnostic: where the staging rings' time goes (slot waits, copies, lock, submits), printed at a prefetch's end |
| `GGML_SYCL_HC_POST_V2` | 1 (service; engine 0) | hc_post one work-item per (dim, token) for all hc outputs (same sums) |
| `STRATA_ROPE_INPLACE` | 1 (service; engine 0) | prompt path: q's RoPE and the attention output's de-RoPE in place (ggml-sycl then rotates only the 64 rope dims) |
| `GGML_SYCL_DSV4_MQA_QV`, `_SM` (`_V2` = both) | 1 (service; engine 0) | the DSV4 MQA prompt attention: Q conversion four floats a work-item; the softmax a sub-group a row |
| `STRATA_XE_CALL_TL`, `STRATA_BATCH_TRACE` | 0 | diagnostics: a prompt call's timeline per card (ms into the call at the card threads' own waits); a served prompt's steps outside the layer pipeline |
| `STRATA_LOOKUP_TP` | 1 (service; engine 0) | serve: lookup drafts (`STRATA_SERVE_LOOKUP` 14) capped to the TP core's 7-row pass - a wider pass took the old dense path (~0.16 s a pass) |
| `STRATA_XE_TP_EVICT` (`_KEEP_GIB` 0.25, `_MEM_GIB` 4, `_PAR` 1, `_ROUND` 1), `STRATA_XE_TP_DEFER_FLEX_GIB` 0.8 | 1 (service; engine 0) | the TP cards' future KV room beyond 0.8 GiB holds experts (each card's last slabs) until a long context needs it; then the slabs it takes go back (their experts read from the model file into host memory - ahead, below), the cards side by side (`_PAR`), the driver waited for once a round of slabs (`_ROUND`) |
| `STRATA_TP_AHEAD_S` | 30 | serve: once a long prompt's batches left look shorter than this, the evictions its KV replicas will need are read ahead (0 = off) |
| `STRATA_XE_TP_AHEAD` (`_READERS` 2, `_CPUS`, `_MEM_GIB` 8, `_WAIT_S` 60) | 1 | the tier's side of it: the slabs' experts read on a helper thread by `_READERS` threads (on `_CPUS`, else the process's CPUs) while `_MEM_GIB` of MemAvailable stays over; an eviction waits for a read at most `_WAIT_S` |
| `STRATA_LI_CHUNK_MIB` | 512 | the prompt indexer's [cap, n] tensors in column slices of at most this many MiB (1M context: ~9 GiB a span otherwise) (not with `STRATA_LI_TOPK`: the op slices its own scratch) |
| `STRATA_SERVE_PARK_MAX` | 262144 (service; engine 65536) | positions a parked conversation may save |
| `STRATA_SERVE_PARK_SLOTS` | 2 | conversations kept beside the live one (1 = the old single park) |
| `STRATA_XE_PIN_SKIP_GIB` | 0 | the first GiB of the ranked arena stay pageable (18 measured worse) |
| `STRATA_DENSE_LAZY_KV` (`STRATA_DENSE_KV_INIT` 262144 engine / 65536 service) | 1 (service since 2026-10-27 23:08; engine 0) | card 0's KV state mapped as the context grows (the other cards pull those rows through a staging buffer: `patch-tpstage.py`) |
| `STRATA_XE_HOTRING` (`_KB` 1024, `_N` 4 engine / 2 service, `_NOSTAGE` 1, `_CHECK` 0) | 1 (service; engine 0) | prompt: pageable arena experts onto the cards through small pinned rings that stay in the L3 (each chunk DMA'd as soon as it is copied), the prefetch's stagers queue their own copies, no pinned layer stages (their budget goes to the arena); `_CHECK=1` compares every streamed / prefetched expert with its source |
| `STRATA_QKV_FUSE` | 1 (service; engine 0) | each layer's q_a + kv as one weight (one mat-vec / GEMM) |
| `STRATA_CONT_SKIP` | 1 | no ggml_cont (a copy op) of tensors that are contiguous already |
| `STRATA_XE_GROUTE` (`_CHECK`), `STRATA_XE_ARM_GRAPH` | 0 | routing on the dense card (a route kernel; the cards pull their jobs) / armed sequences as recorded graphs: correct, no gain measured |
| `GGML_SYCL_FASTMV_SG` / `_UPL` | 4096 / 2 | fastmv's split-K choice: the sub-groups a launch should reach / the units of 32 each lane keeps at least |
| `STRATA_TP` (`STRATA_TP_CACHE` dir) | 4 (service; engine 0) | N > 0: decode / verify passes of <= 7 tokens on engine1 over the first N cards (see Measured, 2026-10-23); its Q8_0 -> Q6_K shard cache (`/srv/dsv41/tpcache`, 5.7 GB); 0 = the old path |
| `STRATA_TP_PROMOTE_PAR` / `_CHUNKS` / `_INORDER`, `STRATA_TP_PC_THREADS` | 8 / 64 / 0, 2 | engine1's promotion cache: jobs a card in flight / copy chunks a pass / copies on the TP queues between passes instead of the tier's own / copy helpers (4 without the card repack) |
| `STRATA_TP_LAZY_KV`, `STRATA_TP_ROOM_KEEP_GIB` | 1, 0.25 | engine1: the other cards' KV replicas mapped as the context grows (their VRAM lent to the flex region meanwhile) / free VRAM a mapping leaves before the flex region gives some back |
| `STRATA_TP_PC_DEVREPACK` | 1 | engine1: promotions copy the expert's file layout and the card repacks it (0: the host repacks, 10-20 ms an expert) |
| `STRATA_TP_PAGE` | 1 (service; engine 0) | engine1's pager: card 0's copies of the weights the TP cards hold (q_b, wo_a, wo_b, shared expert, Engram projection: 4.9 GiB) paged in per layer for prompts, from the shards; the VRAM goes to card 0's expert cache; prompt passes of <= 7 tokens run on the TP core |
| `STRATA_TP_PAGE_SLOTS`, `STRATA_TP_PAGE_PREFETCH` | 3, 1 | the pager's slots (layers mapped at once) / the next layer's copies started while a layer computes |
| `STRATA_TP_PAGE_CHECK`, `_KEEP`, `_DUMMY` | 0 | pager diagnostics: refill every unit at enable and compare with the load / keep the load pages and refill in place (the traffic without the remapping) / with `STRATA_TP_PAGE=1`, no paging - a busy kernel of N us on the dense queue per layer (timing only) |
| `STRATA_TP_PC_EARLY`, `STRATA_TP_PC_PUBLISH` | 1, 1 | engine1: promotions in time for the next pass - a pageable CPU-pool expert's bounce filled right after its CPU use (the slot chosen between passes), finished promotions published before each pass |
| `STRATA_XE_PC_STREAM`, `STRATA_TP_PC_PACE` | 1, 0 | the bounce fills with streaming stores / paused while a TP pass's CPU pool computes (measured slower) |
| `STRATA_TP_ENG_ASYNC` | 1 | engine1: a pass's Engram rows read by a helper thread while the host takes the CPU-pool layers from layer 0 |
| `STRATA_TP_ROUTE_PAR` | 1 | engine1: route_jobs a lane a pick (0: one work-item; same jobs) |
| `STRATA_TP_NJ_DEV`, `STRATA_TP_JOB_GRID`, `STRATA_XE_DOWN_SPLIT` | 1, 8, 0 | engine1's expert kernels: stop at the pass's real job count / gate/up launched for 8 job slots, looping over the jobs (0: a slot a pick) / down's jobs split over its sub-groups (measured slower) |
| `STRATA_TP_MV_TUNE`, `STRATA_TP_MV_FORM` | 1, - | engine1: each Q6_K mat-vec shape's kernel form per pass width timed at load (fastmv's dispatch, dp4a row blockings, XMX K splits; ~2 s) / one form for every width >= 2 (A/B) |
| `STRATA_TP_STAMP_LAYER`, `STRATA_TP_STAMP_TPU` | off, 2790 | diagnostics: clock stamps after every step of one layer (-1: every layer) inside the pass graph, per 128 passes each card's step times / clock ticks a us |
| `STRATA_XE_USAGE_OUT` | off | diagnostics: per routed expert the TP passes that ran it, its picks, card and bytes -> a file at exit (`bench/engine1/place.py`) |
| `STRATA_XE_REPLICA_GIB`, `STRATA_XE_REPLICA_USAGE` | 0.25 (service; 2.5 until 2026-10-24 22:25; engine 0), replica-usage.txt | engine1: card 0's copies of each layer's hottest expert-card experts (pinned host sources, uploaded at every seed) / the uses that choose them (a `STRATA_XE_USAGE_OUT` file; none: the ranking) |
| `STRATA_XE_REPLICA_MARGIN_GIB`, `_PER_LAYER`, `_CHECK` | 1.0, 16, 0 | the copies leave this much of card 0 free / at most this many a layer / compare each copy with its card's bytes |
| `STRATA_TP_DRAFT` | 1 (service; engine 0) | engine1's draft: DSpark's stack on the TP cards (loaded right after the TP core, before the expert tier places experts; card 0's expert cap grows by the ggml stack's VRAM less its share); 0 = the ggml draft on card 0 |
| `STRATA_TP_DRAFT_CHECK`, `STRATA_TP_DRAFT_STAMP` | 0, 0 | diagnostics: the ggml stack beside it, every draft compared (agreement by slot per 64) / clock stamps after each phase of a draft step on every card |
| `STRATA_TP_PC_DEVREPACK`, `STRATA_TP_PC_DMA`, `STRATA_TP_PC_THREADS` | 0, 1, 2 | engine1's promotions: 1 = the card repacks (kernels beside the passes and the draft) / the chunks on the copy engine (0: copy kernels) / fill helpers |
| `STRATA_XE_CPU_REPACK16`, `STRATA_XE_REPACK_FAST` | 1, 1 | the host repack 32 blocks at a time with streaming stores (0: per field) / the card repack with coalesced stores for Q3_K and Q2_K (0: a block a work-item) - both byte-identical (`test_xe_q2k`) |
| `STRATA_TP_ZC_IN` | 1 | engine1: the passes' inputs by a kernel reading the pinned block (0: a copy-engine memcpy, which waits behind the promotions' copies) |
| `STRATA_HOST_CPUS`, `STRATA_XE_POOL_WORKERS`, `STRATA_XE_HELPER_CPUS` | 0, 6, 7,15 (service; engine unset, --threads, unset) | host placement (larry's CPU n and n + 8 are one core): the TP pass's host loop pinned while a pass runs / the CPU pool's workers (cores 1..n) / the promotion fill and Engram helpers' CPUs (unset: the process's original mask) |
| `STRATA_TP_DRAFT_MK`, `STRATA_TP_DRAFT_PUSH`, `STRATA_TP_DRAFT_MK_CHECK` | 1, 1, 0 | engine1's draft: the fused Markov chain / its cross-card sums by P2P store kernels (copies: 0) / the old chain beside the fused one, compared every draft |
| `STRATA_TP_BALANCE`, `STRATA_TP_BAL_JOB_US` / `_GBS` / `_TOK_US` / `_CPU_US` / `_CPU_GBS` / `_SHEXP_US` | 1, 19 / 837 / 3 / 85 / 30 / 10 | engine1's route_jobs: a copied expert's job to the holder that finishes first (0: its owner) / the cost model (a job, bytes, tokens; card 0's CPU-pool wait; a shared-expert block) |
| `STRATA_TP_BENCH_REPLICAS` | 0 | tp_bench: every k-th expert off rank 0 also held by rank 0 |
| `STRATA_XE_CPU_LOG`, `STRATA_XE_CPU_TWICE` | off, 0 | diagnostics: the CPU-pool experts of TP passes, their promotion events and the pass timestamps -> a file (`bench/engine1/cpulog.py`, `pcsim.py`) / each layer's CPU experts run a second time with their weights hot (timing only) |
| `STRATA_DUMP_COL` | -1 | debug: per layer one prompt column's routing, FFN input and routed sum -> `logs/dump_col.<pid>.bin`; `bench/engine1/test_ds41_dumpref.cpp` (experts) and `routercheck.py` (router) recompute them from the GGUF |
| `STRATA_XE_FILL_SLAB` | 1 | placement fills each expert card's last 1 GiB slab past the budget (allocated anyway; 0: the old budget cut) |
| `STRATA_TP_CAP_MIN`, `STRATA_TP_PRERECORD`, `STRATA_TP_SHARE` | 1024, 1, 1 | engine1: the smallest compressed-row bucket of its graphs; graphs recorded at attach; rank 0 reads the dense half's planar weights in place |
| `STRATA_TP_PROF`, `STRATA_TP_SPINSTAT`, `STRATA_TP_PC_DEBUG`, `STRATA_TP_PUSH`, `STRATA_TP_PROMPT` | 0 | engine1 diagnostics: pass phases per 128 passes, cross-card spin counts, the promotion pipeline; partial pushes by kernel (untested); TP passes inside a prompt (test only) |

## Safety (what went wrong while building it, and the rule that follows)

- **Pinned memory**: a 64 GiB pinned arena made Level Zero refuse kernels, and a pinning probe run on top of a running
  engine made larry unresponsive until it rebooted (2026-10-19 ~06:00). TTM caps GPU-driver host memory at ~60.7 GiB
  (`/sys/module/ttm/parameters/pages_limit`); the tier pins 24 GiB and stages pageable copies through small pinned bounces.
  The watchdog kills the server at 54 GiB of driver-held RAM - it did once on 2026-10-21 (58 GiB during a 16K prompt):
  most of that RAM was not pinned memory but UR / UMF peer residency of SYCL device allocations (~1 GiB per GiB of
  VRAM here) plus a flex region larger than the prompt allowed; both fixed (native allocation, flex sized per prompt).
- **VRAM over-commit**: the xe driver spills past a full card into host RAM instead of refusing; card 0 once reached 98 %
  with 15 GiB of experts. Card 0's expert share is capped (10 GiB, 8 GiB reserve).
- **Peer writes into remapped virtual memory**: once card 0 remaps a Level Zero virtual address to other pages, another
  card's copy into that address still lands in the old pages (`bench/engine1/vmm_remap.cpp`: every time). The pager's
  first deploy (2026-10-23 05:20) garbled long prompts that way, overwriting other layers' weights and the expert
  cache. Card 0 now pulls its fills on its own queue; any card writing into another card's VMM must not outlive a remap.
- **NPM**: its custom `server_proxy.conf` uses literal `proxy_pass http://vllm-a:8081` lines, which pin "vllm-a" to the IP
  it had when NPM loaded; after the reboot NPM held 172.23.0.2 itself and the public host answered 502. `fwd-ctl.sh up` and
  `larry-mode small` now reload NPM's nginx (graceful).
- **Killing an engine with copies in flight** faulted a card's copy engine ("CAT error class=bcs"); the xe driver's
  reset recovered it. One engine at a time: `run.sh` holds `/run/lock/ds41-oneshot.lock`; stop the service under
  `/run/lock/larry-profile-a.hold` first, and never `pkill -f` a pattern that matches your own ssh command.
- **Kernels writing another card's memory** (peer writes from cards 1-3 into ggml-sycl's buffers on card 0) gave "Engine
  memory CAT error class=ccs" on all three (2026-10-20); a per-card submission thread faulted card 0's copy engine once and
  the next engine hung until the driver reset card 0. Both are removed: move data between cards with queue memcpys only,
  submit from one thread. After a fault: check `dmesg | grep "CAT error"`, run `bench/submit_bench` and `bench/copy_probe`
  on every card before the service goes back on.
- **Shared USM + a second submitting thread = deadlock** (2026-10-20, the pipelined batch): host writes into
  `malloc_shared` memory go through the driver's page-fault (signal) handler, which waited on a driver mutex held by
  another thread inside `urEnqueueUSMMemcpy`. The tier has no shared USM now (job tables are device memory filled by a
  queued copy from pinned memory). While a pipelined batch runs, every enqueue of every thread (dense half included)
  takes the tier's one submit lock (`Ds41XeTier::submit_mutex`, priority to the dense half); waits stay outside it.
- **Prefetch buffers over-committed a card** (2026-10-20 14:40, OUT_OF_DEVICE_MEMORY then a bcs CAT error): they are
  now sized after run_big's own need for the call's tokens, with a small overflow stream buffer. After a hang, a
  `kill -9` with copies in flight faults the copy engines and the next engine hangs at its first event: reset every
  card's GT (`cat /sys/kernel/debug/dri/0000:{15,1b,22,26}:00.0/gt0/force_reset_sync`), then `bench/copy_probe` each.
- **ggml-sycl's lazy reorders and views**: a weight read through views (wo_a) must never get the whole-tensor reorder
  (the load-time pass skips it); the fused shared-expert weight's halves likewise.
- **A full card mid-session** (2026-10-23 21:41, the first replica try): card 0's copies had taken most of the headroom the
  dense half keeps (`STRATA_XE_DENSE_RESERVE_GIB`); a 1.9 GiB ggml-sycl pool growth in the fourth request of the test
  sequence failed, the engine aborted with copies in flight (copy-engine CAT errors on three cards) and the restarted
  engine hung in its start - GT reset of all four, copy_probe fine. One-shot runs never reach that growth. ggml-sycl's
  allocations now ask a reclaim hook once (`ggml_backend_sycl_set_reclaim_hook`; the tier drops the copies) before they
  fail. Anything borrowing the dense card's headroom must be droppable this way, and must pass `tpg.sh` first.
- **Allocating after the tier's placement takes a card's prompt reserve** (2026-10-24, caught in one-shot logs): engine1's
  draft first loaded with the ggml one (at the first cache seed), after cards 1-3 were placed - its 1.2 GiB a card came
  out of the 3.5 GiB kept free for prompt batches (0.26-0.5 GiB left: the 21:41 pattern waiting for a long prompt). It
  loads right after the TP core now; anything new on the expert cards must be allocated before the tier places them.
- After any reboot `larry-default-layout.service` switches to the default layout (`strata`); `ds41` is accepted there too.

## Not done yet / next

- 2026-10-29 18:00: done - the morning's thinking loops (nine of ~24,600 tokens, 38 % of an agent session's engine
  time) checked against the build (the 10-28 changes give the same logits at 96K / 150K within the run-to-run floor) and
  guarded (`"reasoning_loop_recovery": "wrap"`). Next, decode at depth (49 % of that session, 84.7 tok/s at 40-155K): the
  promotion cache's room (the expert cards keep ~1.3-1.5 GiB free between prompts; slots sized to the largest expert,
  14.5 MiB, against an average CPU-pool expert of 11.7), and the static experts the KV rows took back.

- 2026-10-28 05:10: done - long contexts, part 2: a fresh 967K prompt 338.9 -> 308.3 s (its top-k in two stages; the
  indexer of layers 2 / 8 / 14 as one op whose queries the expert cards share), the first token after it 352.6 -> 308.9 s
  with part 1. Next on long prompts: the expert cards now bound part of it (their expert calls and the indexer's share
  compete: the dense half waits 20 s longer) - the indexer's work made cheaper there (int8 dot products of the FP4
  values: a ~2x ceiling, less a rescale every 32 values), or card 0's own constant part (weight GEMMs ~51 s, attention ~42 s,
  key gather ~31 s of the 967K prompt - optimization #3's).

- 2026-10-28 02:50: done - long contexts, part 1: the first reply after a long prompt (131K 2.2 -> 0.35 s, 967K 8.1 ->
  0.7 s: the expert cards' evictions read ahead and side by side, the replica rows copied once). Next, part 2: the 1M
  prompt itself - its indexer and top-k (~86 of ~347 s): the top-k in two stages first (a row's block maxima, then the
  radix select over the blocks that can hold its 512 best only - the same set), then the indexer (bf16 keys read as they
  are, int8 on the FP4 values, or over the expert cards).

- 2026-10-27 19:20: done - concurrent sessions, stage 1: two conversations of up to 1M at once, taking turns (Strata's
  batch-slot protocol on the TP core's per-conversation state); a re-rendered tail rewinds the live conversation; card
  0's lazy KV map's DEVICE_LOST found (a peer copy across two VMM allocations) and fixed. Next, in order: why the lazy map
  garbles a 970K conversation in a slot that held another one (until then the state is allocated at load, +3.1 GiB on
  card 0); batched decode measured (one pass with both slots' rows - stage 2, built only if it pays); windows inside a
  long prompt's batch (today the other session waits for the whole batch, up to ~6 s).

- 2026-10-27 17:30: done - concurrency stage 0 (the parked conversations as a pool of 8 in 4 GiB, the live one parked
  before another is taken back, two resume bugs). Next, in order: card 0's KV state mapped by the readers' cap buckets
  (`STRATA_DENSE_LAZY_KV` died at 524,288 because the scans read whole power-of-two buckets past what was mapped), then
  stage 1 - two resident sessions of up to 1M taking turns pass by pass (per-slot state on the four cards, the TP core's
  graphs keyed by slot, the serve loop speaking Strata's batch-slot protocol the server already has), then a measurement
  of batched decode. After it: short tier calls, big prompts on both sides, the long-context indexer.

- 2026-10-25 12:05: done - layers 0-19's attention through oneDNN's fused SDPA (fresh 16K -0.2 s, 31K -0.5 s).
  Next: the tier calls now bound layers 0-19 at 16K as well as 8K (their GEMM phase ~2x xe_dqgemm alone; the first call
  of a layer waits for its staging), the key gather (still 0.7 ms a span: keys written and read again), the regrow
  after the batch (0.3-0.5 s before the first token).

- 2026-10-25 11:00: done - the staging measured to its limits (DRAM ~33 GB/s, one user copy engine a card) and its
  pinned share and small uploads fixed (fresh 8K -0.2 s, 16K -0.3 s, 31K -0.2..-0.7 s). Next: card 0's span at 16K+
  (its compute engine is the bound; the attention ~40 % of it: a fused kernel that keeps every sub-group busy - the
  first one idled them between four barriers a chunk); at 8K the calls' GEMM phase (~2x xe_dqgemm alone).

- 2026-10-25 07:00: done - the dense span's glue kernels (fresh 16K ~9.5 -> ~8.6-9.0 s, 31K ~16.3 -> ~14.9-15.3 s).
  Next on card 0's span: the HC mixes (rms_norm over the 4-stream state + a [24 x 20480] GEMM, 2 x 0.3 ms a span: one
  pass), the attention's key gather (0.68 ms) and its two GEMM passes over K (a fused kernel needs a better tiling
  than fattn-dsv4-pf.hpp's), then the attention block over the cards.

- 2026-10-25 05:15: done - the dense card out of the prompt calls' critical path (fresh 8K ~7.1 -> ~6.7 s, 16K ~10.35
  -> ~9.6 s, 31K ~17.5 -> ~16.3 s, agent replay 118.6 -> 115.6 s). Next: layers 0-19 are bound by card 0's dense
  parts (~125 ms a half at 16K) - split the attention block (q_b, attention, wo_a, wo_b: ~3.3 ms of a ~7.8 ms span) over
  the cards (KV replicas and the TP shards exist; the expert cards have ~30 ms a half to spare), then the rest of the
  dense work; `begin_batch` (0.2-0.33 s a batch: host embeddings / maps, init graphs a span at a time); the upper
  layers' staging rate (~16 of the ring's ~21.7 GB/s).

- 2026-10-25 02:30: done - fresh prompts 16K 13.4-14.1 -> 10.1-10.2 s, 31K 19.9-23.1 -> 17.2-17.6 s (the Engram rows,
  keep-cache for every batch, the first-call stall, fp16 parts), and the card-0 faults behind keep-cache shrinks found
  and fixed. Next (the layers now): in layers 0-19 the tier's calls (~140-150 ms a half: the FFN input copy 17 ms,
  ~21 overflow experts streamed in the call, GEMMs ~25 ms) bound card 0's dense parts (~120 ms) - the FFN input in
  fp16, the overflow prefetched, then card 0's dense work (its attention: ~2.1 ms of a 7 ms span) and the split over
  the cards; layers 20-39 are bound by the prefetch staging (~120 ms a layer).

- 2026-10-24 23:05: done - two park slots (a stale parked conversation no longer costs the next side request's main
  conversation a re-read). Next: why card 0 faults when its KV state is mapped after load; the TP prompt path for long
  prompts from empty (card 0's dense half bounds them: 15K 11 s, 31K 20 s); evicted experts back onto their cards.

- 2026-10-24 22:30: done - a 1M-token context in the service (984K needle found, decode 62 tok/s at that depth), at
  the 262K config's speed for ordinary lengths since card 0's cache took the replicas' VRAM. Next: why card 0 faults
  when its KV state is mapped after load (then 1M would cost no cache at all); evicted experts back onto their cards
  when no long conversation is left; two park slots; the replay's decode at 115 vs 123 tok/s (the replicas).

- 2026-10-24 18:10: done - an agent session replay (`bench/speed/agentx.py`) and the expert copies of prompts through rings in the L3 (`STRATA_XE_HOTRING=1`: 13.0 -> 21.7 GB/s for pageable experts; the replay's prompts 127 -> 116.5 s, 31K needle 1605 -> 1702 tok/s); a park saves only its conversation's rows. Next: the user asked for a 1M-token context (KV ~3 KB a token: ~0.78 GiB a card at 262K, ~3.1 GiB at 1M - measuring the cost and long prompts at depth); two park slots (a stale parked conversation blocks the next park); `STRATA_XE_PREFETCH_MIN=2560` (3-4K turns at depth -0.25 s, within the noise so far).

- 2026-10-24 15:22: done - lookup drafts on the TP core (`STRATA_LOOKUP_TP=1`): a copy-heavy edit 62 -> 183-191 tok/s,
  math +27 %, svcbench 86 -> 95; no prefetch for batches ending before 8K tokens + keep-cache to 8K: prompts from empty
  4.4K -25 %, 6K -18 %, 8K -11 %. A bigger promotion cache (FLEX 2.5) did not pay. Next: the agentic session's CPU-pool
  experts (22-54 a pass with the wide lookup passes; measure again without them - `STRATA_XE_CPU_LOG` on an agent
  replay), turns of <= 128 new tokens run as passes (52-124 tokens took 1-2 s, ~60 tok/s: a CPU-light batch may beat it,
  `STRATA_SERVE_BATCH_MIN`), the pool's activation quantisation, card 0's re-seed after prompts over 8K tokens (~0.9 s).

- 2026-10-24 10:57: done - short prompts' light experts on the CPU pool (`STRATA_XE_CPU_LIGHT=4`): pp512 +36-47 % at
  any depth (speed table), 160 tokens +37 %; the prefetch from 4,096 tokens (long prompts' upper layers from 2,048,
  3-4K prompts layers 20-39 only): 2-4K-token prompts +10-29 % more; long prompts unchanged (a card-0 VRAM regression
  of the first build found by the speed table and fixed). Next, in order: no prefetch for shallow batches below ~8K
  tokens (+14 % at 6K, drafted: `STRATA_XE_PREFETCH_DEPTH`; never for calls over ~5K columns - card 1 ran out of VRAM);
  the pool's activation quantisation (~3-10 ms a call, one thread); card 0's cache seed after long prompts (0.84 s at
  the card's PCIe rate: overlap it with the first decode passes); decode's CPU-pool experts (math: ~22 a verify pass,
  ~9 ms of ~38 - the pinned arena holds card 0's cache, so promotions and the pool see pageable experts).

- 2026-10-24 05:30: done - the standard speed table (llama-bench's pp512 / tg128 at 0-96K, `vllm bench serve` on
  MT-Bench; see the top). Not run yet: the quality benchmarks DeepSeek publishes for this model (DeepInfer's
  `eval/`: AIME 2026 99.9 %, GPQA Diamond 93.4 %, HLE 42.6 % Pass@1) - their prompts and rule scorer through this API,
  hours to a day of service time at one sample a problem.

- 2026-10-24 03:10: done - the CPU pool's threads placed (its layers ~12-14 % faster; cycle -1.6 / -2.6 %) and the draft's
  Markov chain fused with pushed sums (exact; essay -1.8 %).  The pool's kernels are at the DRAM ceiling; what holds it
  back now is the promotions' DRAM traffic beside it (~30 MB a promotion).  Next, in order: fewer bytes per promotion or
  promotions timed away from the CPU layers; fewer CPU experts (math: ~30 a pass); an adaptive replica set.

- 2026-10-24 01:55: done - DSpark's draft on the four cards (`STRATA_TP_DRAFT=1`): a draft step 4.9 -> ~3 ms, the
  essay's cycle -5..-7 %, code -9 %, math noisy; promotions without kernels (the host writes the planes).  Next, in
  order: the CPU pool (math: ~14 CPU-pool layers a pass, ~8 ms of a 33 ms cycle; the fills now share DRAM with it);
  a low-priority queue for what promotions still run on the cards; the draft's Markov chain (5 sequential cross-card
  sums); an adaptive replica set.

- 2026-10-23 22:20: done - card 0's copies of each layer's hottest expert-card experts, handed out per pass by
  route_jobs (essay -2 %, code verify -3 %, math -6 %), with ggml-sycl's reclaim hook (the first try crashed on a full
  card 0).  Next, in order: an adaptive copy set (the tier's counts of recent requests instead of a file); the CPU
  pool's per-layer latency (~455 us for one expert, ~370 of it the bytes); the draft (5-6 ms a cycle on card 0).

- 2026-10-23 20:35: done - engine1's verify kernels: route_jobs a lane a pick, the expert kernels bounded by the real
  job count, Q6_K mat-vec forms tuned per shape and width (exact XMX for q_b / wo_b / shared-expert down): a 5-token
  pass -3.9 %, the DSpark cycle ~-2 %.  The cards are busy all the pass; ~17-27 % of it is cards waiting for each
  other (card 0 for the CPU pool, the rest for the slowest card).  Next, in order: replicas of each layer's hottest
  experts on card 0 with per-pass balancing in route_jobs; the CPU pool's per-layer latency (~455 us for one expert,
  ~370 of it the bytes); then the draft (5-6 ms a cycle on card 0).

- 2026-10-23 17:31: done - engine1's promotions in time for the next pass (bounce filled at the CPU use, copies during
  the draft, published before the pass): CPU-pool experts a verify pass -29 %, a DSpark cycle -6 % at the same pass
  widths.  The PCIe plan measured out: 94-95 % of the CPU pool's experts are pageable.  Next, in order: the draft's
  remaining +0.5 ms under TP; Engram rows read ahead during the draft; more promotion slots (card 0 has ~2 GiB free in
  decode) and keeping them through batch prompts; the first uses (~40 % of the CPU pool's experts: a faster CPU part
  or routing prediction); TP prefill and the draft on the TP core.
- 2026-10-23 06:02: done - engine1's pager in the service (`STRATA_TP_PAGE=1`): card 0's copies of the sharded dense
  weights paged in per layer for prompts, card 0 pulling the fills (other cards' copies into remapped virtual memory
  land in the old pages); essay +4-7 %, short prompts ~2x, long prompts even.  Row 111 of p2k: a chaotic row, not a
  race.  Next, in order: the CPU-pool experts of verify passes (9-11 ms of a TP verify pass; pinned arena experts
  computed by idle cards over PCIe where a layer has several); the draft's remaining +0.5 ms under TP; Engram rows read
  ahead during the draft; the pass's own kernels (a unitrace of `tp_bench`); TP prefill and the draft on the TP core.
- 2026-10-23 ~02:35: done - engine1 in the service (`STRATA_TP=4`): KV replicas mapped as the context grows,
  promotions repacked on the cards, the head split on TP's slices, full slabs (both paths).  Essay +16 % through the
  service; at 95K TP and today's path are even (VRAM).  Next, in order: card 0's copy of the other ranks' weight shards
  (~3.8 GB: only the prompt path reads them - page them in per layer from cards 1-3, then card 0 takes ~380 more
  experts); the CPU-pool experts of verify passes (9-11 ms of a TP verify pass; pinned arena experts computed by idle
  cards over PCIe where a layer has several); the draft's remaining +0.5 ms under TP; Engram rows read ahead during
  the draft; the pass's own kernels (a unitrace of `tp_bench`); TP prefill and the draft on the TP core.
- 2026-10-23 early hours: done - engine1, the N-card tensor-parallel decode / verify core (`STRATA_TP=N`, off): plain
  decode +34 %, DSpark +5 % (verify -12 %), PPL unchanged; not deployable at 262K yet (VRAM).  Next, in order: KV
  replicas grown to the context reached instead of sized for ctx (~0.85 GB a card at 262K), TP's weights inside the
  tier's VRAM budget so the service's prompt path keeps its buffers; then the CPU-pool experts in verify passes (7-9
  ms of a TP verify pass), the draft's extra ~1.1 ms under TP, the pass's own kernels (a unitrace of `tp_bench`), and
  TP prefill (drops card 0's full dense copy, 6.5 GB) and the draft on the TP core.
- 2026-10-22 evening: done - fused decode kernels (attention core, int8 handoffs, in-place outputs, indexer, hc merge,
  finish fold): plain decode +8-10 %, DSpark verify -10 %, bit-exact op by op.  Then measured (commit 50257caa's
  knobs): moving the shared expert off card 0 would NOT pay for the service - run twice it costs plain decode 4.7 %
  and DSpark verify nothing (the expert cards / CPU pool bound verify passes); card 0's own expert jobs ~0.8 %.  The
  real post-readback cost is the CPU-pool experts: ~550-620 us each, all on the critical path - ~10 % of a plain run,
  ~18 % of a DSpark run - with the pool's kernels at 21-26 GB/s against larry's 38-40 GB/s of DRAM reads (more workers
  do not help; SMT siblings make it worse).  Options: split each CPU-pool expert between the CPU and an idle card over
  PCIe (only the pinned 20.8 GiB of the 68.6 GiB arena DMA at full speed; ~3-4 % plain, ~5-7 % DSpark), a stronger
  route bias (1.0: DSpark ~+8 % at KL 0.041 vs exact), or the four-card engine (TP degree a setting, any card count).
  Smaller fusions left: the router mat-vec + top-k, wo_b's activation quantisation inside wo_a, the attention's two
  kernels, one readback instead of three, the compressor (layers 2/8/14/20) and Engram (1/14) chains.
- 2026-10-22 late morning: done - the routing bias measured, the quant checked against the original, the new engine's
  step 0 (gate not met: 355 us vs 300).  Next: real fused kernels for the small-op chains (attention's ~14, the
  indexer's top-k chain, the ffn's router / job prep) in the current engine - pays in the service directly and
  measures what the four-card engine would reach; then decide on the four-card engine (another ~1.4-1.5x).  Quality:
  exact routing (`--route-bias 0`) for non-code / Chinese work at -29 % speed; a less aggressive quant needs RAM larry
  does not have (~20 GiB spare).
- 2026-10-22: done - the decode / verify lightning indexer on XMX (long contexts ~1.6x through the service), per-job
  expert kernel forms, q_a + kv as one weight, routing on the GPU (opt-in, no gain yet).  Next: with `STRATA_XE_GROUTE`
  only the CPU-pool experts still need the router readback - a helper thread that waits for it and signals the sum
  through a host-signalled event (`bench/gr/hs_probe`: works as a dependency, ~3-6 us) lets the main thread queue
  layer l+1 before layer l's readback, the host headroom the multi-card splits (`STRATA_TP_OPROJ`, the shared expert on
  cards 1-3) need to pay; a per-shape split-K rule for fastmv (~4-10 us a layer); for prompts, card 0's copies (3.6 s
  of a 31.6K prompt: expert uploads and the X sends).
- 2026-10-21 night: done - utilization accounting, the attention output split, XMX experts (both opt-in), Engram pool.
  Next, in order of expected gain for the service (DSpark): a GPU-side expert dispatch (no host round trip between the
  router and the cards - the cards start ~60 us after a verify readback now, and it lets the host run ahead, which the
  splits need); int8 XMX experts (flat in tokens at dp4a speed: wider verify passes then cost little); the shared expert
  on cards 1-3 riding along with the routed experts; then more of the dense half split (q_b) once the host is ahead.
- 2026-10-21 evening: done - prompt stalls (wo_a, fp16 weight cache, Engram), the DSpark draft. Next: a GPU-side expert
  dispatch (no host round trip between the router and the cards' start: ~20-25 us of every decode / verify layer),
  the CPU-pool experts in verify (~40 us a layer), the draft's remaining unfused HC / shared expert (~0.3 ms a pass),
  batch 1's cold Engram reads in long prompts.
- 2026-10-21 afternoon: done - the fused hyper-connection ops. Next for decode: the dense queue's late view of the expert
  cards' parts (25-110 us a layer, cause unknown - not the SYCL dependency mechanism, see Measured); the CPU-pool
  experts in verify passes (~40 us a layer); the DSpark draft (7.6 ms of a ~47 ms pass). For prompts: more pinned
  arena (each GiB saves ~2 GiB of staging traffic a pass; TTM caps the box at ~60 GiB, the watchdog at 54 hidden), or
  a routing-aware prefetch (predict layer l+1's token counts from layer l's state; stream only experts with many).
- 2026-10-21 (the utilization round, see Measured): next for decode - fuse the hyper-connection ops (post / prenorm
  mat-vec / mixes / Sinkhorn / pre / norm: ~34 us of ~8 kernels a layer; DeepInfer's "mega mhc" does it in one per
  sub-layer), GPU-side expert dispatch (the host's readback -> release round trip, ~25 us a layer). For prompts: an
  opt-in DeepInfer-style YoCo truncation (layers 20-39 on the last ~140-512 tokens instead of the exact 2560 reach:
  an approximation - measure first); the expert tier's per-batch arena stream is near its PCIe / host-memory floor.
- DeepInfer (github.com/deepseek-ai/DeepInfer, DeepSeek's own V4.1-Flash engine): vLLM-based, TileLang + FlashMLA /
  DeepGEMM / DeepEP kernels for SM100 and Ascend 950, the whole model in VRAM across 4-64 GPUs, no RAM / CPU expert
  tier. Not a fit to fork for larry (the model does not fit 4 x 32 GB; every kernel would need a SYCL rewrite); its
  ideas are listed above. Clone (read-only): /opt/ai/ext/DeepInfer.

- 2026-10-20 night: the pass ring (commit 4ec2f186) - card 0 68 -> 76 % busy over a 31.6K prompt, 1013-1082 tok/s one-shot,
  1114 through the service. Found with gdb stack samples: the small uploads spun in the driver's CPU-copy path.
- 2026-10-20 evening, the prompt half of the doubling round (commits ae41dac9, 49987d86): fused key gather, fp16 prompt
  GEMMs, expert prefetch, the pipelined batch, card-to-card X, 16K batches, DSpark kept resident. Next for prompts: card 0
  is the critical path again (~8-9 s of an 8.5K prompt, ~23-25 s of 31K) - its attention runs as two batched GEMMs over
  materialised f16 keys (a gather-free XMX flash-attention kernel would drop the key buffer), and the expert tier
  dequantises each expert twice per layer now (once per column half; a dequantising GEMM would read the quants directly).
- 2026-10-20: the doubling round (commits b9f2199d, dd987504, 723969ce, 7f4d8161, ee389c08): see Measured. Profiling notes:
  unitrace (`/opt/src/pti-gpu/tools/unitrace`, `utrace.sh`, `utan.py`, `utgap.py`) shows card 0's kernels; replayed
  graphs hide theirs. `tl.sh`/`tm.sh` run the stamped timeline; `bench/fm2/fastmv_bench_nc` the mat-vec variants.
- Next for decode: card 0's ~45 small kernels a layer (norms, hyper-connection mixes, KV gathers, quant_x) as fused
  kernels; an XMX (DPAS) multi-column Q6_K kernel for verify passes; fewer arena misses (a dynamic VRAM cache - card 0
  cannot take more than ~10 GiB of experts safely); the DSpark draft (~8.4 ms/pass, mostly weight traffic).
- Larger: split the dense half's mat-vecs across the four cards (tensor parallel) - the cards' bandwidth is idle, but
  every layer would add two cross-card exchanges.

Older notes (2026-10-19):


- Done 2026-10-19 (ggml-sycl patch): the prompt's flash attention (DeepSeek's one-query-per-batch MQA with sinks) and its
  lightning indexer run as batched fp16 oneMKL GEMMs on XMX (8.5K prompt 201 -> ~310 tok/s, 31.7K 260 -> ~430 tok/s).
- Prompt: what is left is building each query's key set (gather/concat/cast, ~5 s of an 8.5K prompt), the dense GEMMs in
  fp32 (~4 s; `GGML_SYCL_F16` gave -13 %), and streaming arena experts over the shared PCIe uplink.
- Decode's ~45 ms/token of dense half is ~16 small matvecs a layer (ggml-sycl issues `wo_a`'s 8 groups as 8 copies + 8
  matvecs) and ~1,000 small kernels; the expert tier is ~11 ms/token (kernels at ~165 GB/s: a repacked, wide-load layout
  is the next step).
- `--kv-fp4` (FP4 compressed KV, +1 GiB VRAM at 512K) needs MXFP4 keys in ggml-sycl's lightning indexer.
- A fused quantised-weight XMX GEMM for the prompt's experts (now: fp16 dequantisation + oneMKL).
- Vision (the sidecar carries the vision tower; the Intel path is text-only). *Done 2026-10-26: pictures through the
  service - see "Measured (2026-10-26 22:40 - 23:45: pictures ...)".*
