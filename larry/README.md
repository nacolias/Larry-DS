# Larry-DS: DeepSeek-V4.1-Flash-1017 on four Intel Arc Pro B70s

This branch (`larry-xe`) of [Xolotl](https://github.com/Indras-Mirror/Xolotl) serves
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). That is 40 layers of 384 routed
experts (top-6), with Engram n-gram memory and a 1M-token context. It runs as the 163 GiB GSQ quant on **Larry**:
four Intel Arc Pro B70 (32 GB) cards, a Ryzen 7 5800X, 121 GiB of DDR4 and two NVMe drives. It serves one request
at a time through an OpenAI- and Anthropic-compatible API.

Xolotl's `ds41` engine was built for one NVIDIA card. This branch adds the Intel side:

- **A multi-card expert tier** (`tools/v41/engine/xe/`). The routed experts go hottest first into the four cards' VRAM
  (about 9,500 of 15,360). The rest sit in a RAM arena (24 GiB of it pinned). In decode, a 6-thread CPU pool
  computes the arena's experts. In prompts, the pool takes the lightly used ones and the rest stream onto the cards,
  the CPU writing them straight into VRAM through the cards' 32 GiB BARs.
- **A tensor-parallel decode core**, "engine1" (`tools/v41/engine/tp/`, design in `tp/DESIGN.md`). Decode and
  speculative-verify passes (up to 7 tokens) split the dense half over all four cards by heads and columns. Each card
  replays a SYCL graph per pass, and the KV state is replicated as the context grows. The DSpark draft (the
  checkpoint's own MTP heads) is split the same way.
- **A ggml-sycl patch** (`tools/v41/engine/xe/ggml-sycl-dsv41.patch`, applied to llama.cpp at Xolotl's pin). It adds:
  - Xolotl's fused DeepSeek-V4 ops on SYCL;
  - dp4a and XMX mat-vec kernels;
  - the lightning indexer on XMX;
  - layers 0-19's prompt attention through oneDNN's fused SDPA;
  - an AVX2 Q2_0 dot for the CPU pool;
  - an op profiler (off by default).
- **Engine changes** (`tools/v41/engine/ds41_*.cpp`):
  - A pipelined prompt path: layer-major batches of up to 16K tokens, each layer's arena experts prefetched while the
    layer before it runs.
  - The keep-cache: a prompt batch borrows only the VRAM it needs from card 0's expert cache, sized from the last
    batches, and the cache refills in the background.
  - A reader for a paged copy of the Engram tables.

[`ENGINEERING-LOG.md`](ENGINEERING-LOG.md) is the build log, as kept on Larry. It records every change with its A/B
numbers, what went wrong, and the rules that followed.

## Results (Larry, 2026-10-24 to 10-26)

**Service check** (`checks/tsvc2.sh` through the server; 2026-10-26 04:30-04:40, the current build):

| check | result |
| --- | --- |
| fresh needle prompts (prompt tok/s) | 8K 1,531 then 1,745, 16K 2,283, 31K 2,589 - all four passphrases found |
| chat replies (4 prompts x 300 tokens, temperature 1, DSpark) | 94.6 tok/s, 91.8 after the long prompts |
| essays (3 x 800 tokens) | 69.7 tok/s, 69.1 after the long prompts |
| tool-call round trip | correct |
| host memory held by the GPU driver | 33 GiB at most (the watchdog acts at 54) |

Decode speed varies from run to run (sampling, draft acceptance, what the cards hold). Over the last 14 checks, the
first chat pass ran 94.6-102.7 tok/s and the first essay 69.7-81.3 tok/s. In every check the first 8K needle, which
follows the decode tests, is slower than the second.

**Fresh prompts from an empty context** (A/B windows abwT94, abwT86, abwT102 in the log, 2026-10-25/26: a server with
the service's settings on a side port, greedy):

| prompt tokens | 530 | 1,021 | 2,003 | 3,003 | 4,519 | 8,054 | 16,099 | 31,008 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seconds | 1.96 | 2.10 | 2.81 | 3.49 | 4.02 | 4.66 | 7.00 | 12.09 |

**Follow-up turns** in a replayed agent session (synthetic, `bench/speed/agentx.py`; context 8K-38K; abwT103):
- 56 new tokens: 0.47-0.62 s.
- 500: 1.83 s.
- 1.4K: 2.66 s.
- 3.7K: 3.9 s.
- 6.5K: 4.46 s.
- Agent replies: 121 tok/s (greedy).

**Long contexts** (2026-10-24, an earlier build; prompt then decode):

| context | prompt | decode after it |
| --- | --- | --- |
| 129K | 80 s (1,610 tok/s) | 90 tok/s |
| 490K | 393 s | 81 tok/s |
| 984K | 1,140 s (863 tok/s) | 61.7 tok/s |

**Fidelity against DeepSeek's own release** (2026-10-22). The original FP8/FP4 model was teacher-forced on Larry's
CPU with DeepSeek's `inference/model.py`, and compared on 36,846 tokens across 18 x 2,048-token sequences:

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
| model | [kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata) at `7aa29158` (see Provenance) |

## Quickstart

It is built and tuned for exactly this machine: four 32 GB B70s (AOT-compiled for `bmg-g31`), at least 128 GB of
RAM, and about 360 GiB of NVMe (460 with the paged Engram copy), ideally split over two drives. Other card counts
and GPUs are untested.

1. **Prerequisites.** The `xe` driver must bind all four cards with full-size BARs (`lspci -vv` shows `[size=32G]`).
   Install the GPU runtime and oneAPI packages above, plus `build-essential cmake ninja-build git python3-venv
   python3-dev`. The serving user needs the `render` group and unlimited locked memory (`LimitMEMLOCK=infinity` in
   the unit).
2. **Clone and build.** `bootstrap.sh` does the following:
   - fetches llama.cpp at Xolotl's pin and applies both ggml patches;
   - installs the deploy scripts and checks into `DS41_HOME` (default: the checkout's parent, `/opt/ai/dsv41`);
   - creates the venv and builds `ds41_generate_xe` + `libds41_xe.so`.

   ```bash
   git clone -b larry-xe https://github.com/nacolias/Larry-DS.git /opt/ai/dsv41/xolotl
   /opt/ai/dsv41/xolotl/larry/deploy/bootstrap.sh     # DS41_SRV=/srv/dsv41: the second drive; NINJA_JOBS=N if RAM is short
   ```
3. **Download the model** (163 GiB GGUF, 5.3 GiB DSpark sidecar, tokenizer, `rank.bin`, two 95 GiB Engram tables),
   at the revision Larry runs:
   ```bash
   /opt/ai/dsv41/dl.sh main     # -> /opt/ai/dsv41/models/gsq1017 (~263 GiB)
   /opt/ai/dsv41/dl.sh l14      # -> /srv/dsv41/gsq1017: layer 14's Engram table (~95 GiB); one drive: DS41_SRV=/opt/ai/dsv41/models
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
   - As a service: `deploy/host/` has `ds41.service` and its memory watchdog `ds41-watch.service`, the GPU clock pin
     (`gpu-clock.sh`: the cards' minimum clock at max while the server runs) and the watchdog script. Install steps
     are in `ds41.service`'s header.
   - The first start writes the TP shard cache (`/srv/dsv41/tpcache`, 6 GiB). A normal start takes 80-95 s to the
     first request.
7. **Check it**: `/opt/ai/dsv41/tsvc2.sh` (as in Results; a few minutes). The checks read `DS41_URL` and `DS41_API_KEY` /
   `DS41_KEY_FILE` (`checks/ds41api.py`).

## Settings

- **`ds41-xe-engine.sh`** holds every setting the service runs with, as environment defaults. A variable already set
  wins. Each comment gives the A/B result that set the value.
- **`larry-ds41.json`** holds the engine flags (`--ctx 1048576 --prompt-batch 16384 --route-bias 0.5`
  `--route-bias-file 0.3`), the sampling defaults (temperature 1.0, top-p 0.95), and the model paths (filled in from
  `larry-ds41.json.in`).
- `ENGINEERING-LOG.md` ("Knobs") lists the tier's knobs.

The Larry-specific ones:
- `STRATA_HOST_CPUS`, `STRATA_XE_POOL_WORKERS` and `STRATA_XE_HELPER_CPUS` follow the 5800X's core layout.
- `STRATA_XE_PINNED_GIB=24` stays under the xe driver's TTM limit on GPU-driver host memory (~60.7 GiB on Larry). Never
  pin past it: Larry hung once that way.
- The VRAM budgets (`STRATA_XE_DENSE_GIB`, `_RESERVE_GIB`, `_FLEX_GIB`) assume 32 GB cards.
- Larry also binds the server to its docker bridge (`DS41_HOST=172.23.0.1`), reads its key from a shared env file
  (`DS41_KEY_FILE`), checks the switch's uplink (`DS41_UPLINK_PCI=0000:00:03.1`), and keeps `"api_monitor": true`.
  `api_monitor` keeps the last 100 prompts and answers in memory for `/api-monitor`, so the template ships it off.

## Known limits

- **One request at a time** (first in, first out). A side request can park one long conversation.
- **One engine per box.** The service holds every card and ~35 GiB of driver-held host memory. `run.sh` refuses to
  start beside it.
- **The first prompts after a start are slower.** Their first calls load oneMKL's kernels and compile oneDNN's
  attention partitions.
- **Runs are not bit-reproducible.** Split sums, sampling and the cache state vary.
- **Based on Xolotl `f7c561b7` (2026-10-18).** Upstream has moved on since, and those commits are not merged here.
- **Four cards only.** A split for any card count is planned, not built.

## Provenance

The GGUF's metadata calls it "DeepSeek V4.1 Flash UNCENSORED" and names dealignai's abliterated FP8 repo as its
source. The weights say otherwise. All 40 layers' router weights are bit-identical to `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
The token embeddings match. The attention-output and shared-expert projections differ only by Q8_0 rounding (0.54-0.64 %).
There is no rank-1 edit: the difference's top singular value is 0.13 or less, against 12.0 for dealignai's. So this is
DeepSeek's official 1017 release, and the metadata is mislabeled. The scripts are in `bench/provenance/`, and the
details are in `ENGINEERING-LOG.md` ("Model provenance").

## Layout

| path | what |
| --- | --- |
| `tools/v41/engine/xe/` | the Intel expert tier (`ds41_xe_tier.cpp`), its kernels, `xe.cmake`, the ggml patch, tests |
| `tools/v41/engine/tp/` | engine1, the tensor-parallel decode core, and its `DESIGN.md` |
| `tools/v41/engine/ds41_*.cpp` | the engine: prompt pipeline, keep-cache, Engram reader, serve-path changes |
| `larry/deploy/` | bootstrap, build, download, the engine and server launchers, config template, `host/` units |
| `larry/checks/` | the service check (`tsvc2.sh`) and its pieces (needle, decode, essay, tool call), `tok.py` |
| `larry/bench/` | the A/B harness (`speed/`), microbenchmarks, quality and provenance scripts ([README](bench/README.md)) |
| `larry/patches/` | the scripts each change was made with: a record, not a build step ([README](patches/README.md)) |
| `larry/ENGINEERING-LOG.md` | the log |

## Credits and license

- [Xolotl](https://github.com/Indras-Mirror/Xolotl) and [Strata](https://github.com/Niko1221/Strata) (MIT) provide the
  engine, the server and the DeepSeek-V4.1 support this builds on.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT) provides the SYCL backend.
- DeepSeek made the model.
- kaushikvira made the GSQ quant.
- Larry's changes are under the repository's MIT license.
