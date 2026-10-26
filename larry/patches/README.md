# larry/patches - how each change was made

Each script made one change on Larry: a source patch to the engine, the tier or ggml (in `third_party/llama.cpp`),
or a service setting (`svc-*.py`, against `ds41-xe-engine.sh` / `larry-ds41.json`). Each is anchored on exact text,
and most check for their own marker first.

They are the record, not a build step. They were applied in turn, and what survived is already here:
- engine changes in this branch's commits;
- ggml changes in `tools/v41/engine/xe/ggml-sycl-dsv41.patch`;
- settings in `larry/deploy/`.

Some were experiments left off or reverted; `ENGINEERING-LOG.md` says which. Each header says what the script
changed and why, with the numbers.
