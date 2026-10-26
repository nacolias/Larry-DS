// stub (local syntax checks only): the one oneMKL call the tier makes
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
namespace oneapi::mkl {
enum class transpose : char { nontrans, trans, conjtrans };
namespace blas::row_major {
template <typename... A> sycl::event gemm(sycl::queue&, A&&...) { return {}; }
}  // namespace blas::row_major
}  // namespace oneapi::mkl
