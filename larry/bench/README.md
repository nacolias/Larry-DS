# larry/bench - the measurement harness and probes

Everything here was written on Larry and runs there as written. The scripts assume `/opt/ai/dsv41` (the checkout at
`/opt/ai/dsv41/xolotl`), the service on `172.23.0.1:18080` (an A/B server on `:18081`), and the API key in
`/opt/docker/vllm/.env`. Adjust those to reuse them. Run outputs, logs and datasets are not included.

| dir | what |
| --- | --- |
| `speed/` | the A/B harness: `abw.sh` (a window: waits for the service to be idle, stands in on its port, runs each config, brings the service back), `tabx.sh` / `abx.py` (fresh prompts, edits, decode), `agentx.py` (a synthetic agent session: the repo's own source and fixed assistant turns), `opprof.sh` (card 0's per-op profile, `GGML_SYCL_OP_PROF=1`), `pipesum*.py` / `calltl*.py` (pipeline and call timelines), `nicebuild.sh` (a build frozen while the service has requests) |
| `quality/` | AIME 2026 and GPQA-Diamond runners (`aime.py`, `gpqa.py`; `mkdata.py` / `mkgpqa.py` build the inputs, which are not included) |
| `qv/` | the quant-vs-original comparison: DeepSeek's FP8/FP4 release teacher-forced on the CPU (`qv_teacher.sh`), the engine's logits on the same rows (`qv_engine.sh`), `qv_score.py`; `qv_score.json` / `teacher_summary.json` are that run's results |
| `provenance/` | the weight checks against the official and the dealignai repos (`prov2.py`, `prov3.py`; `prov3.log` is the result) |
| `engram/` | `engram_epg.py` (the paged Engram copy: convert / verify), `iops.c` (random-read rates), `test_engpg.cpp` |
| `conc/` | concurrent sessions: `s0test.sh` (a private server: needles, the tool call and `reusetest.py` - conversations rotated through the park pool, the partial resume) |
| `vision/` | pictures: `venc_test.py` (one picture through the encoder, started as the server starts it), `vtest.sh` / `vtest.py` (a test server with the `"vision"` section: two cards through both APIs, a follow-up that reuses the picture, a second picture that must not) |
| `dqb/`, `fa/`, `fm*/`, `gr/`, `fuse/`, `xe/` | SYCL kernel probes and benchmarks: dequant GEMMs, flash attention, fast mat-vec, staging rings, copies, P2P |
| `tp/`, `engine0/`, `engine1/` | the tensor-parallel core's probes and benchmarks (`engine1/vmm_remap.cpp`: the remapped-VMM peer-write hazard) |
| `*.cpp` at the top | first-day probes (launch, submit, copy, pin, dp4a, clock); `early/`: the first days' one-shot runners |
