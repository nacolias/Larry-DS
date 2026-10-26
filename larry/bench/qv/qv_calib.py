#!/usr/bin/env python3
"""qv_calib.py ARGS... - Xolotl's v41_calib.py main() unchanged, with the one name it uses but never imports
(ABL, an ablation manifest module: cmd_layer's summary calls ABL.manifest_id()) stubbed as "no ablation"."""
import sys
sys.path.insert(0, "/opt/ai/dsv41/xolotl/tools/v41")
import v41_calib as C  # noqa: E402


class _NoAblation:
    @staticmethod
    def manifest_id():
        return None


if not hasattr(C, "ABL"):
    C.ABL = _NoAblation()

# load_block opens every file a layer's tensors live in - for layers 1 / 14 also the Engram tables' file (the engine's
# extracted engram-L14.safetensors has 48 bytes of alignment padding between its two tensors, which safetensors'
# strict header check rejects: "invalid offset").  The tables are not Block parameters (MemmapEngramEmbedding reads
# them through their offsets), so load_block gets the weight map without them; cmd_layer's own lookup keeps them.
_orig_load_block, _orig_weight_map = C.load_block, C.weight_map


def _load_block(blk, L):
    C.weight_map = lambda: {k: v for k, v in _orig_weight_map().items() if ".engram.embed." not in k}
    try:
        return _orig_load_block(blk, L)
    finally:
        C.weight_map = _orig_weight_map


C.load_block = _load_block
C.main()
