#!/bin/bash
# bootstrap.sh - set the DeepSeek-V4.1 Xe engine up from this checkout (idempotent): llama.cpp at Xolotl's pin with
# Xolotl's ggml patch plus larry's (tools/v41/engine/xe/ggml-sycl-dsv41.patch), the deploy scripts and checks
# installed into DS41_HOME, the Python venv, the build.  DS41_HOME defaults to the directory holding this checkout
# (larry: /opt/ai/dsv41, the checkout at /opt/ai/dsv41/xolotl):
#   git clone -b larry-xe https://github.com/nacolias/Larry-DS.git /opt/ai/dsv41/xolotl
#   /opt/ai/dsv41/xolotl/larry/deploy/bootstrap.sh
# Needs oneAPI 2026.1 (ONEAPI_VARS) and intel-ocloc; the model download is separate (dl.sh, the top-level README.md).
#   DS41_SRV (/srv/dsv41): the second NVMe drive - layer 14's Engram table, the paged Engram copy, the TP cache
#   BUILD=0: install only, no build
set -euo pipefail
X=$(cd "$(dirname "$0")/../.." && pwd)
D=${DS41_HOME:-$(dirname "$X")}
SRV=${DS41_SRV:-/srv/dsv41}
KIT=$X/larry
mkdir -p "$D/logs"
[ "$X" = "$D/xolotl" ] || [ -e "$D/xolotl" ] || ln -s "$X" "$D/xolotl"   # every script finds the checkout at $D/xolotl
cd "$X"
P=$PWD/tools/v41/engine/xe/ggml-sycl-dsv41.patch
# larry's patch goes on top of xolotl-ggml.patch, so fetch-llama.sh (which re-applies that one unless it alone reverses)
# runs only for a tree without both
if [ -d third_party/llama.cpp/.git ] && git -C third_party/llama.cpp apply --reverse --check "$P" 2>/dev/null; then
  echo "ggml: third_party/llama.cpp already has both patches"
else
  bash tools/xolotl/fetch-llama.sh                     # llama.cpp at the pin + xolotl-ggml.patch
  git -C third_party/llama.cpp apply "$P" && echo "ggml-sycl patch: applied"
fi
grep -q 'larry: the Intel Arc tier' tools/v41/engine/v41_engine.cmake ||
  echo 'include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/xe/xe.cmake OPTIONAL)   # larry: the Intel Arc tier' >> tools/v41/engine/v41_engine.cmake
# the kit into $D: a file already there that differs is kept, the kit's copy goes beside it as .kit-new
put() {   # put SRC DST [MODE]
  local tmp
  tmp=$(mktemp "$2.XXXX")
  cp "$1" "$tmp"
  case "$1" in *.in) sed -i "s|@DS41_HOME@|$D|g; s|@DS41_SRV@|$SRV|g" "$tmp" ;; esac
  chmod "${3:-644}" "$tmp"
  if [ ! -e "$2" ]; then mv "$tmp" "$2"; echo "installed $2"
  elif cmp -s "$tmp" "$2"; then rm -f "$tmp"
  else mv "$tmp" "$2.kit-new"; echo "kept $2 (differs from the kit: see $2.kit-new)"; fi
}
for f in build-xe.sh dl.sh export-patch.sh run.sh ds41-xe-engine.sh ds41-serve.sh ds41-vision.sh; do put "$KIT/deploy/$f" "$D/$f" 755; done
put "$KIT/deploy/xe-env.sh" "$D/xe-env.sh"
put "$KIT/deploy/replica-usage.txt" "$D/replica-usage.txt"
put "$KIT/deploy/larry-ds41.json.in" "$D/larry-ds41.json"
put "$KIT/checks/tsvc2.sh" "$D/tsvc2.sh" 755
for f in ds41api.py needle.py svcbench.py essay.py tooltest.py tok.py; do put "$KIT/checks/$f" "$D/$f"; done
mkdir -p "$SRV/tpcache" "$SRV/gsq1017" 2>/dev/null || echo "bootstrap: could not create $SRV/tpcache - make it, or set DS41_SRV" >&2
cd "$D"
[ -x venv/bin/python ] || { python3 -m venv venv && venv/bin/pip -q install -U "huggingface_hub[hf_xet]" -r xolotl/requirements.txt; }
[ "${BUILD:-1}" = 0 ] || ./build-xe.sh
