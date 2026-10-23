// tools/v41/engine/tp/tp_fastmv.cpp - ggml-sycl's fastmv mat-vecs compiled into libds41_tp in their own namespace
// (tpfmv): the engine links ggml-sycl statically, so the two copies' host functions and SYCL kernel names stay apart
#define fastmv tpfmv
#include "fastmv.cpp"
