# Larry-DS: DeepSeek-V4.1-Flash-1017 on four Intel Arc Pro B70s

This repository serves [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) from
one workstation, **Larry**: four Intel Arc Pro B70 (32 GB) cards, a Ryzen 7 5800X, 121 GiB of DDR4 and two NVMe
drives. The model has 40 layers of 384 routed experts (top-6), Engram n-gram memory and a 1M-token context. It runs as
the 163 GiB GSQ quant through an OpenAI- and Anthropic-compatible API, pictures included.

It is a fork of [Xolotl](https://github.com/Indras-Mirror/Xolotl), whose `ds41` engine runs this model on one
NVIDIA card. This fork adds the Intel side: SYCL / Level Zero code that spreads the model over four Arc cards, system
RAM, the CPU and two NVMe drives. Everything was measured on Larry, one A/B change at a time, between 2026-10-19 and
10-27. [`larry/ENGINEERING-LOG.md`](larry/ENGINEERING-LOG.md) is that record.

## Headline numbers

| | Larry, 2026-10-26 |
| --- | --- |
| fresh 8K-token prompt | 4.48 s |
| fresh 31K-token prompt | 10.5 s (2,946 tok/s) |
| follow-up turn of 500 new tokens at 38K context | 1.6 s |
| chat replies (temperature 1, speculative decoding) | 95-103 tok/s |
| agent-session replies (greedy) | 120-124 tok/s |
| 984K-token prompt | 6 min, then 67 tok/s |
| context window | 1,048,576 tokens |
| a new 800x600 picture | 7-8 s (~6 s of it the CPU encode); a follow-up question on it 0.2-1.2 s |
| perplexity vs DeepSeek's own FP8/FP4 release | 1.15x overall (code 1.06x, math 1.08x) |

Through the server, fresh 31K-token prompts went from 1,673 tok/s on Oct 23 to 3,017 tok/s on Oct 26 (~3,020 when
repeated warm). The full tables
are under [Results](#results).

## How it works

| | holds | runs |
| --- | --- | --- |
| card 0 | the dense half (attention, router, shared expert, Engram projection, head; Q8_0 requantized to Q6_K at load), an expert cache of ~19 GiB (~1,800 experts) | the dense half of prompts; its share of every decode pass |
| cards 1-3 | ~24 GiB of experts each, their share of the dense half and the KV state, a 1 GiB flex region, the draft's share | their experts; their share of every decode pass |
| RAM | the arena: the other experts, ~80 GiB (24 GiB of it pinned) | a 6-core CPU pool for arena experts |
| NVMe | the Engram tables, 2 x 95 GiB (one per drive), plus a 98 GiB paged copy of layer 1's | ~48 small random reads a token |

About 9,500 of the 15,360 routed experts live in VRAM, placed hottest first.

- **Decode is speculative and tensor-parallel.**
  - DSpark (the checkpoint's own MTP heads) drafts up to 4 tokens, plus lookup drafts from the context. A draft
    window is cut where the drafts' combined confidence drops below 0.5 (Project Maya's rule, +3 % decode).
  - "engine1" (`tools/v41/engine/tp/`) verifies the drafts in one pass of up to 7 tokens over all four cards. The dense
    half is split by heads and columns, and each card replays a SYCL graph.
  - Arena experts a pass needs run on the CPU pool. Hot ones are promoted into the flex regions for later passes.
- **Prompts are pipelined.**
  - They run layer-major, in batches of up to 16,384 tokens. Card 0 runs the dense half while the expert cards
    compute the routed experts.
  - Batches of 3,072 tokens or more (once the context is past 5,000 tokens) get each layer's arena experts staged
    onto the expert cards ahead of the layer. Pinned ones go by DMA; pageable ones the CPU writes straight into VRAM
    through the cards' 32 GiB BARs.
  - Card 0's expert cache lends a batch only the VRAM it needs, sized from the last batches, and refills in the
    background.
- **Pictures.** DeepSeek's vision tower (in the DSpark sidecar GGUF: a 1024-wide ViT and the aligner, 485M parameters)
  encodes a picture on the CPU ([`larry/deploy/ds41-vision.sh`](larry/deploy/ds41-vision.sh) around Xolotl's
  `ds41_vision.py`; ~6 s an 800x600 picture, 2.2 GB of RAM) into its span of image tokens. The engine places the rows,
  routes them with the model's VL bias and gives them no Engram, as DeepSeek's `model.py` does; OpenAI `image_url`
  parts and Anthropic image blocks both work, and a conversation reuses a picture it has already read.
- **Kernels.** [`ggml-sycl-dsv41.patch`](tools/v41/engine/xe/ggml-sycl-dsv41.patch) is applied to llama.cpp at
  Xolotl's pin. It adds:
  - Xolotl's fused DeepSeek-V4 ops on SYCL;
  - dp4a and XMX mat-vec kernels;
  - the lightning indexer on XMX, a prompt's as one fused kernel whose score matrix never leaves the registers;
  - the prompt's hyper-connection mixes in one pass over the 4-stream state, wo_a's 8 slices as one product fed by
    the attention's fp16 output;
  - layers 0-19's prompt attention through oneDNN's fused SDPA;
  - an AVX2 Q2_0 dot for the CPU pool;
  - an op profiler (off by default).

The multi-card expert tier is [`tools/v41/engine/xe/ds41_xe_tier.cpp`](tools/v41/engine/xe/ds41_xe_tier.cpp), and
engine1's design is [`tools/v41/engine/tp/DESIGN.md`](tools/v41/engine/tp/DESIGN.md).

## Results

**Service check** ([`larry/checks/tsvc2.sh`](larry/checks/tsvc2.sh) through the server; 2026-10-26 ~23:20, the
current build):

| check | result |
| --- | --- |
| fresh needle prompts (prompt tok/s) | 8K 1,422 then 1,899, 16K 2,506, 31K 3,185 - all four passphrases found |
| chat replies (4 prompts x 300 tokens, temperature 1, DSpark) | 105.2 tok/s, 102.4 after the long prompts |
| essays (3 x 800 tokens) | 80.6 tok/s, 78.0 after the long prompts |
| tool-call round trip | correct |
| pictures ([`larry/bench/vision/vtest.py`](larry/bench/vision/vtest.py): two test cards, OpenAI and Anthropic APIs) | both described right (shapes, colours, the text on them); 7-8 s with the encode, a follow-up on the same picture 0.2 s |
| host memory held by the GPU driver | 34 GiB at most (the watchdog acts at 54) |

Decode speed varies from run to run (sampling, draft acceptance, what the cards hold). Over the last 19 checks, the
first chat pass ran 93.3-102.7 tok/s and the first essay 69.7-81.3 tok/s. In every check the first 8K needle, which
follows the decode tests, is slower than the second.

**Fresh prompts from an empty context** (A/B windows abwT94, abwT86, abwT102 and, from 8K up, abwT133 in the log,
2026-10-25/26: a server with the service's settings on a side port, greedy):

| prompt tokens | 530 | 1,021 | 2,003 | 3,003 | 4,519 | 8,054 | 16,099 | 31,008 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seconds | 1.96 | 2.10 | 2.81 | 3.49 | 4.02 | 4.48 | 6.39 | 10.53 |

**Follow-up turns** in a replayed agent session (synthetic, [`larry/bench/speed/agentx.py`](larry/bench/speed/agentx.py);
context 8K-38K; abwT124, with the fused hc path since turned off - ~1 % slower without it):
- 56 new tokens: 0.48-0.59 s.
- 500: 1.61 s.
- 1.4K: 2.41-2.48 s.
- 3.7K: 3.74-3.82 s.
- 6.5K: 3.95 s.
- Agent replies: 120-124 tok/s (greedy).

**Long contexts** (needles from empty, prompt then decode; 131K / 256K: abwT130, 2026-10-26 with the fused hc path,
as now; 973K: abwT144, the current config; 490K: Oct 24's build):

| context | prompt | decode after it |
| --- | --- | --- |
| 131K | 41 s (3,192 tok/s; Oct 24: 80 s) | 86 tok/s |
| 256K | 74 s (3,466 tok/s) | 82 tok/s |
| 490K | 393 s (Oct 24) | 81 tok/s |
| 973K | 346 s (2,815 tok/s; Oct 24: 1,140 s) | 67 tok/s; an 11.7K-token turn on it 10-15 s (once out of card 0's memory before `STRATA_SERVE_KEEP_LEARN_DEPTH_GIB`) |

**Fidelity against DeepSeek's own release** (2026-10-22). The original FP8/FP4 model was teacher-forced on Larry's
CPU with DeepSeek's `inference/model.py`, and compared on 36,846 tokens across 18 x 2,048-token sequences
([`larry/bench/qv/`](larry/bench/qv/)):

| | code | math | agent | English | Chinese | all |
| --- | --- | --- | --- | --- | --- | --- |
| perplexity ratio | 1.06 | 1.08 | 1.11 | 1.15 | 1.44 | 1.15 |
| top-1 = original's | 92 % | 92 % | 87 % | 78 % | 75 % | 85 % |

That is about a 2-2.5-bit quant's loss overall, smallest on code and math.

## What Larry runs

| | |
| --- | --- |
| GPUs | 4 x Intel Arc Pro B70 32 GB (BMG-G31, `8086:e223`), 32 GiB resizable BARs, behind one PCIe 4.0 x16 switch |
| host | AMD Ryzen 7 5800X (8 cores), 121 GiB DDR4, NVMe: Sabrent 1 TB (`/opt`) + WD SN850X 1 TB (`/srv`) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.2.9 (self-built, `-bmg1`), the in-kernel `xe` driver |
| kernel command line | `xe.force_probe=* i915.force_probe=* pci=realloc=off pci=disable_acs_redir=pci:8086:e2f0;pci:1000:c010` (ACS redirection off on the switch ports, for card-to-card copies) |
| GPU runtime | `intel-opencl-icd` / `libze-intel-gpu1` / `intel-ocloc` 26.35.39758.17, `libze1` / `libze-dev` 1.32.0 (Launchpad PPA `kobuk-team/intel-graphics-testing`) |
| oneAPI | DPC++ 2026.1.1, oneMKL 2026.1.0 (SYCL BLAS), oneDNN 2026.0.2 (`intel-oneapi-compiler-dpcpp-cpp-2026.1`, `intel-oneapi-mkl-devel-2026.1`, `intel-oneapi-dnnl-devel`) |
| build | cmake 4.2, ninja 1.13, Python 3.14 |
| model | [kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata) at `7aa29158` (see [Provenance](#provenance)) |

## Quickstart

It is built and tuned for exactly this machine: four 32 GB B70s (AOT-compiled for `bmg-g31`), at least 128 GB of
RAM, and about 360 GiB of NVMe (460 with the paged Engram copy), ideally split over two drives. Other card counts
and GPUs are untested.

1. **Prerequisites.** The `xe` driver must bind all four cards with full-size BARs (`lspci -vv` shows `[size=32G]`).
   Install the GPU runtime and oneAPI packages above, plus `build-essential cmake ninja-build git python3-venv
   python3-dev`. The serving user needs the `render` group and unlimited locked memory (`LimitMEMLOCK=infinity` in
   the unit).
2. **Clone and build.** [`bootstrap.sh`](larry/deploy/bootstrap.sh) does the following:
   - fetches llama.cpp at Xolotl's pin and applies both ggml patches;
   - installs the deploy scripts and checks into `DS41_HOME` (default: the checkout's parent, `/opt/ai/dsv41`);
   - creates the server's venv and the image encoder's (`tvenv`: CPU torch, numpy, pillow; `VISION=0` skips it);
   - builds `ds41_generate_xe` + `libds41_xe.so`.

   ```bash
   git clone -b larry-xe https://github.com/nacolias/Larry-DS.git /opt/ai/dsv41/xolotl
   /opt/ai/dsv41/xolotl/larry/deploy/bootstrap.sh     # DS41_SRV=/srv/dsv41: the second drive; NINJA_JOBS=N if RAM is short
   ```
   On 2026-10-26 a clean clone from GitHub went through this on Larry. Its repo and ggml trees came out identical to
   the live ones, and it built in under 10 minutes with `NINJA_JOBS=3` at nice 19. On 2026-10-27 the picture setup
   went through the same way (`BUILD=0` into a scratch `DS41_HOME`, then `dl.sh vision`), and the encoder read a test
   picture through that install.
3. **Download the model** (163 GiB GGUF, 5.3 GiB DSpark sidecar with the vision tower, tokenizer, `rank.bin`, two
   95 GiB Engram tables) at the revision Larry runs, and DeepSeek's reference image code for the encoder:
   ```bash
   /opt/ai/dsv41/dl.sh main     # -> /opt/ai/dsv41/models/gsq1017 (~263 GiB)
   /opt/ai/dsv41/dl.sh l14      # -> /srv/dsv41/gsq1017: layer 14's Engram table (~95 GiB); one drive: DS41_SRV=/opt/ai/dsv41/models
   /opt/ai/dsv41/dl.sh vision   # -> /opt/ai/dsv41/models/ds1017/inference: vision.py + image_processor.py (12 KB)
   ```
4. **Optional: the paged Engram copy** (98 GiB on the faster drive). Each row of layer 1's table becomes one aligned
   4 KiB read instead of two random ones. The engine uses `engram-L1.epg` from the first `--engram` directory that
   has it, and otherwise reads the shards:
   ```bash
   cd /opt/ai/dsv41 && mkdir -p /srv/dsv41/engram-epg
   venv/bin/python -I xolotl/larry/bench/engram/engram_epg.py convert models/gsq1017/engram-layers.1/engram-L1.safetensors 1 /srv/dsv41/engram-epg/engram-L1.epg
   venv/bin/python -I xolotl/larry/bench/engram/engram_epg.py verify  models/gsq1017/engram-layers.1/engram-L1.safetensors 1 /srv/dsv41/engram-epg/engram-L1.epg
   ```
5. **Smoke test** (one-shot, as root, with the service stopped):
   ```bash
   cd /opt/ai/dsv41
   venv/bin/python tok.py enc logs/hello.i32 "Write a haiku about GPUs."
   ./run.sh hello --ids-file logs/hello.i32 -n 64 && venv/bin/python tok.py dec - < logs/hello.out
   grep -E "^(prompt|decode):" logs/hello.err
   ```
6. **Serve.**
   - By hand: `/opt/ai/dsv41/ds41-serve.sh` (127.0.0.1:18080).
   - As a service: [`larry/deploy/host/`](larry/deploy/host/) has `ds41.service` and its memory watchdog
     `ds41-watch.service`, the GPU clock pin (`gpu-clock.sh`: the cards' minimum clock at max while the server runs)
     and the watchdog script. Install steps are in `ds41.service`'s header.
   - The first start writes the TP shard cache (`/srv/dsv41/tpcache`, 6 GiB). A normal start takes 80-95 s to the
     first request.
   - The server also starts the image encoder ([`ds41-vision.sh`](larry/deploy/ds41-vision.sh), 2.2 GB of RAM). It
     needs `tvenv` and `dl.sh vision`; without them the server does not start (see Settings for a text-only config).
7. **Check it**: `/opt/ai/dsv41/tsvc2.sh` (as in Results, ending with a picture, `imagetest.py`; a few minutes). The
   checks read `DS41_URL` and `DS41_API_KEY` / `DS41_KEY_FILE` ([`larry/checks/ds41api.py`](larry/checks/ds41api.py)).

## Settings

- **[`ds41-xe-engine.sh`](larry/deploy/ds41-xe-engine.sh)** holds every setting the service runs with, as
  environment defaults. A variable already set wins. Each comment gives the A/B result that set the value.
- **[`larry-ds41.json.in`](larry/deploy/larry-ds41.json.in)** holds the engine flags (`--ctx 1048576
  --prompt-batch 16384 --route-bias 0.5 --route-bias-file 0.3 --vision`), the sampling defaults (temperature 1.0,
  top-p 0.95), the model paths and the `"vision"` section (the encoder script, the sidecar it reads, 6 CPU threads).
  For a text-only server, delete the section and `--vision`. `bootstrap.sh` renders it as `larry-ds41.json`.
- [`larry/ENGINEERING-LOG.md`](larry/ENGINEERING-LOG.md) ("Knobs") lists the tier's knobs.

The Larry-specific ones:
- `STRATA_HOST_CPUS`, `STRATA_XE_POOL_WORKERS` and `STRATA_XE_HELPER_CPUS` follow the 5800X's core layout.
- `STRATA_XE_PINNED_GIB=24` stays under the xe driver's TTM limit on GPU-driver host memory (~60.7 GiB on Larry). Never
  pin past it: Larry hung once that way.
- The VRAM budgets (`STRATA_XE_DENSE_GIB`, `_RESERVE_GIB`, `_FLEX_GIB`) assume 32 GB cards.
- Larry also binds the server to its docker bridge (`DS41_HOST=172.23.0.1`), reads its key from a shared env file
  (`DS41_KEY_FILE`), checks the switch's uplink (`DS41_UPLINK_PCI=0000:00:03.1`), and keeps `"api_monitor": true`.
  `api_monitor` keeps the last 100 prompts and answers in memory for `/api-monitor`, so the template ships it off.

## Known limits

- **One request at a time** (first in, first out). Up to 8 other conversations wait parked in host memory (at most
  4 GiB together), so moving between chats or agents does not re-read their prompts. Concurrent sessions are the
  next feature (engineering log, 2026-10-27).
- **One engine per box.** The service holds every card and ~35 GiB of driver-held host memory. `run.sh` refuses to
  start beside it.
- **Pictures are encoded on the CPU.** A new 800x600 picture takes ~6 s on 6 threads (a repeat comes from the
  server's cache), up to DeepSeek's 1,024 image tokens a picture. Through the model, only one picture per request has
  been run (both APIs, with follow-ups). Several pictures in a request and pictures in tool results are covered only by
  the server's tests with a stub encoder ([`test_serve_ds41.py`](tools/v41/serve/test_serve_ds41.py)), and pictures
  deep in a long context are untested.
- **The first prompts after a start are slower.** Their first calls load oneMKL's kernels and compile oneDNN's
  attention partitions. Some starts also stalled for seconds: under memory pressure the xe driver's shrinker wrote
  the engine's host buffers out to swap and the first prompts read them back. Since 2026-10-26 the load and the
  Engram reads no longer flood the page cache (`STRATA_XE_LOAD_DIRECT`, `STRATA_ENGRAM_DROP`), which removes most of
  it; swap off while the service runs removes the rest (see the engineering log, 2026-10-26 15:54-19:00).
- **Runs are not bit-reproducible.** Split sums, sampling and the cache state vary.
- **Based on Xolotl `f7c561b7` (2026-10-18).** Upstream has moved on since, and those commits are not merged here.
- **Four cards only.** A split for any card count is planned, not built.

## Provenance

The GGUF's metadata calls it "DeepSeek V4.1 Flash UNCENSORED" and names dealignai's abliterated FP8 repo as its
source. The weights say otherwise. All 40 layers' router weights are bit-identical to `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
The token embeddings match. The attention-output and shared-expert projections differ only by Q8_0 rounding (0.54-0.64 %).
There is no rank-1 edit: the difference's top singular value is 0.13 or less, against 12.0 for dealignai's. So this is
DeepSeek's official 1017 release, and the metadata is mislabeled. The scripts are in
[`larry/bench/provenance/`](larry/bench/provenance/), and the details are in the log ("Model provenance").

## Layout

| path | what |
| --- | --- |
| [`tools/v41/engine/xe/`](tools/v41/engine/xe/) | the Intel expert tier (`ds41_xe_tier.cpp`), its kernels, `xe.cmake`, the ggml patch, tests |
| [`tools/v41/engine/tp/`](tools/v41/engine/tp/) | engine1, the tensor-parallel decode core, and its `DESIGN.md` |
| `tools/v41/engine/ds41_*.cpp` | the engine: prompt pipeline, keep-cache, Engram reader, serve-path changes, picture spans |
| [`tools/v41/serve/`](tools/v41/serve/) | the server's DeepSeek-V4.1 glue (`serve_ds41.py`: prompts, tool calls, special-token escaping, pictures) and its tests |
| [`larry/deploy/`](larry/deploy/) | bootstrap, build, download, the engine and server launchers, config template, `host/` units |
| [`larry/checks/`](larry/checks/) | the service check (`tsvc2.sh`) and its pieces (needle, decode, essay, tool call, picture), `tok.py` |
| [`larry/bench/`](larry/bench/) | the A/B harness (`speed/`), microbenchmarks, quality, provenance and picture (`vision/`) scripts |
| [`larry/patches/`](larry/patches/) | the scripts each change was made with: a record, not a build step |
| [`larry/ENGINEERING-LOG.md`](larry/ENGINEERING-LOG.md) | the log |
| everything else | upstream Xolotl at `f7c561b7`; its README is [`README.xolotl.md`](README.xolotl.md) |

## Credits and license

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) (MIT) provide the
  engine, the server and the DeepSeek-V4.1 support this builds on.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT) provides the SYCL backend.
- DeepSeek made the model.
- kaushikvira made the GSQ quant.
- This fork's changes are under the same [MIT license](LICENSE).
