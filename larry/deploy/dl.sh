#!/bin/bash
# dl.sh main|l14|vision - download kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata at the revision larry runs,
# split across two NVMe drives: `main` (the GGUF, the DSpark sidecar with the vision tower, rank.bin, the tokenizer,
# layer 1's Engram table; ~263 GiB) into DS41_HOME/models/gsq1017, `l14` (layer 14's Engram table, ~95 GiB) into
# DS41_SRV/gsq1017.  One drive: DS41_SRV=DS41_HOME/models.  `vision`: the image encoder's model code (ds41-vision.sh),
# DeepSeek's reference inference/vision.py + image_processor.py (deepseek-ai/DeepSeek-V4.1-Flash-1017 at 1017's
# release; 12 KB) into DS41_HOME/models/ds1017.
D=${DS41_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SRV=${DS41_SRV:-/srv/dsv41}
REPO=kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata
REV=7aa29158172112fcdccb4c519eb31fdc4bb9238c
DSREPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
DSREV=1bae0e1088993b9b82de08f59de5be5a34e42170
export HF_XET_HIGH_PERFORMANCE=1
case "$1" in
  main)   exec "$D/venv/bin/hf" download $REPO --revision $REV --local-dir "$D/models/gsq1017" --exclude "engram-layers.14/*" ;;
  l14)    exec "$D/venv/bin/hf" download $REPO --revision $REV --local-dir "$SRV/gsq1017" --include "engram-layers.14/*" ;;
  vision) exec "$D/venv/bin/hf" download $DSREPO inference/vision.py inference/image_processor.py --revision $DSREV \
            --local-dir "$D/models/ds1017" ;;
  *)      echo "usage: dl.sh main|l14|vision" >&2; exit 2 ;;
esac
