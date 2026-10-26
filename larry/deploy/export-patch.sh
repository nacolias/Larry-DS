#!/bin/bash
# export-patch.sh - larry's ggml patch = third_party/llama.cpp minus (llama.cpp at Xolotl's pin + xolotl-ggml.patch),
# written to tools/v41/engine/xe/ggml-sycl-dsv41.patch (bootstrap.sh applies it after fetch-llama.sh)
set -euo pipefail
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
X=$D/xolotl
L=$X/third_party/llama.cpp
PIN=$(grep -o 'PIN=[0-9a-f]*' $X/tools/xolotl/fetch-llama.sh | cut -d= -f2)
REF=$(mktemp -d $D/llref.XXXX)
git -C $L archive $PIN ggml | tar -x -C $REF
(cd $REF && git init -q && git apply $X/tools/xolotl/ggml/xolotl-ggml.patch)
OUT=$X/tools/v41/engine/xe/ggml-sycl-dsv41.patch
: > $OUT
cd $L
# tracked files that differ from the reference, then untracked sources (larry's new files)
{ git diff --name-only $PIN -- ggml; git ls-files --others --exclude-standard -- ggml; } | sort -u | while read -r f; do
  case "$f" in *.o|*.a|*.so) continue;; esac
  if [ -f "$REF/$f" ]; then
    cmp -s "$REF/$f" "$f" && continue
    diff -u --label "a/$f" --label "b/$f" "$REF/$f" "$f" | sed "1i diff --git a/$f b/$f" >> $OUT || true
  else
    { echo "diff --git a/$f b/$f"; echo "new file mode 100644"; diff -u --label /dev/null --label "b/$f" /dev/null "$f" || true; } >> $OUT
  fi
done
rm -rf $REF
grep '^diff --git' $OUT | sed 's/^diff --git a\///; s/ b\/.*//'
echo "$(wc -l < $OUT) lines -> $OUT"
# check: it applies to the reference
REF2=$(mktemp -d $D/llref.XXXX)
git -C $L archive $PIN ggml | tar -x -C $REF2
(cd $REF2 && git init -q && git apply $X/tools/xolotl/ggml/xolotl-ggml.patch && git apply --check $OUT && echo "patch applies cleanly on pin + xolotl-ggml.patch")
rm -rf $REF2
