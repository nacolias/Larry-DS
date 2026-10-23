// tools/v41/engine/tp/tp_fastmv.cpp - ggml-sycl's fastmv mat-vecs compiled into libds41_tp in their own namespace
// (tpfmv): the engine links ggml-sycl statically, so the two copies' host functions and SYCL kernel names stay apart
#define fastmv tpfmv
#include "fastmv.cpp"

namespace tpfmv {

// larry, engine1: one Q6K_R8 mat-vec in a named form, for the TP core's per-shape choice (fastmv's own dispatch is form
// 0).  1-4: the row-blocked dp4a kernel, 2 / 4 rows a lane group, normal / 256-register mode (a.xq / a.xd); 5-8: the XMX
// kernel with its K split over 1 / 2 / 4 / 8 sub-groups (xT / xdT: quant_xT's layout).  False: a form this shape or
// these inputs cannot take.
bool launch_form(sycl::queue & q, const Args & a, const sycl::half * xT, const float * xdT, int form) {
    if (a.ncols < 1 || a.ncols > 8 || a.K <= 0 || a.N <= 0 || a.K % 256) return false;
    if (form >= 1 && form <= 4 && (!a.xq || !a.xd)) return false;
    if (form >= 5 && (!xT || !xdT || (1 << (form - 5)) > a.K / 256)) return false;
    switch (form) {
        case 0: return launch(q, WType::Q6K_R8, a);
        case 1: return r8m_cols<2, false>(q, a);
        case 2: return r8m_cols<2, true>(q, a);
        case 3: return r8m_cols<4, false>(q, a);
        case 4: return r8m_cols<4, true>(q, a);
        case 5: xmxd::run<1>(q, a, xT, xdT); return true;
        case 6: xmxd::run<2>(q, a, xT, xdT); return true;
        case 7: xmxd::run<4>(q, a, xT, xdT); return true;
        case 8: xmxd::run<8>(q, a, xT, xdT); return true;
    }
    return false;
}

}  // namespace tpfmv
