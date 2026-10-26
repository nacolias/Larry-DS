#!/bin/bash
# ds41-vision.sh - the DeepSeek-V4.1 image encoder for the server's "vision" config section (strata-vision's line
# protocol: READY / ENC <image> <out> / QUIT): Xolotl's tools/v41/vision/ds41_vision.py on the CPU, the sidecar's
# vision tower (--mmproj), DeepSeek's reference vision.py / image_processor.py (DS41_REFERENCE: the inference/ folder of
# deepseek-ai/DeepSeek-V4.1-Flash-1017).  Needs a python with torch (CPU), numpy and pillow: DS41_TVENV (default
# DS41_HOME/tvenv).  The server passes --mmproj, --model and --threads.  ~2.3 GB of RAM; ~6 s an 800x600 picture
# on 6 threads.
D=${DS41_HOME:-/opt/ai/dsv41}
exec "${DS41_TVENV:-$D/tvenv}/bin/python" -I "$D/xolotl/tools/v41/vision/ds41_vision.py" "$@" --device cpu \
  --reference-dir "${DS41_REFERENCE:-$D/qv/ds1017/inference}"
