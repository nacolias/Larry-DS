# larry/ - Larry's deploy kit, checks, benchmarks and log

The overview, results and quickstart are in the [top-level README](../README.md).

| path | what |
| --- | --- |
| [`deploy/`](deploy/) | the scripts, which `bootstrap.sh` installs into `DS41_HOME` (see below) |
| [`deploy/host/`](deploy/host/) | `ds41.service`, `ds41-watch.service`, `gpu-clock.sh` (the cards' minimum clock at max while the server runs), `mem-watch.sh` (kills the server before driver-held RAM runs out) |
| [`checks/`](checks/) | `tsvc2.sh` (the service check) and its pieces, `ds41api.py` (endpoint and key), `tok.py` (prompt <-> token ids) |
| [`bench/`](bench/) | Larry's A/B harness, kernel probes, and the quality, fidelity and provenance scripts ([README](bench/README.md)) |
| [`patches/`](patches/) | the script each change was made with: a record, not a build step ([README](patches/README.md)) |
| [`ENGINEERING-LOG.md`](ENGINEERING-LOG.md) | Larry's build log, 2026-10-19 to 10-26, newest first |

The scripts in `deploy/`:
- `bootstrap.sh`: applies the ggml patches, installs the kit into `DS41_HOME`, creates the venv and builds.
- `build-xe.sh`: the build itself.
- `dl.sh`: downloads the model at Larry's revision.
- `run.sh`: a one-shot engine run.
- `xe-env.sh`: the oneAPI / Level Zero environment and the model paths.
- `ds41-xe-engine.sh`: the engine as the server starts it, with every setting.
- `ds41-serve.sh`: the server.
- `larry-ds41.json.in`: the engine flags and model paths.
- `replica-usage.txt`: the expert-usage counts behind card 0's replicas.
- `export-patch.sh`: re-exports the ggml patch from `third_party/llama.cpp`.

Paths:
- `DS41_HOME` (default: the checkout's parent; on Larry `/opt/ai/dsv41`, the checkout at `/opt/ai/dsv41/xolotl`) holds
  the installed scripts, the venv, `models/` and `logs/`.
- `DS41_SRV` (`/srv/dsv41`) is the second drive: layer 14's Engram table, the paged Engram copy and the TP cache.
- The checks find the server through `DS41_URL` and `DS41_API_KEY` / `DS41_KEY_FILE`.
- `bench/` scripts are as Larry ran them (its paths and ports).
