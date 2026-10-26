// dp4a_probe.cpp - which spelling of an int8 4-way dot product compiles to Xe2's dp4a: plain C++ (dpct style) or
// IGC's builtin.  Times a dependent chain of each and checks they agree.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat) __attribute__((const));
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_us(int c, int a, int b, bool sat) __attribute__((const));
#endif

static inline int dp4a_cpp(int a, int b, int c) {
    const sycl::vec<int8_t, 4> va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>();
    const sycl::vec<int8_t, 4> vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}
static inline int dp4a_us_cpp(uint32_t a, int b, int c) {   // a unsigned bytes, b signed
    const sycl::vec<uint8_t, 4> va = sycl::vec<uint32_t, 1>(a).as<sycl::vec<uint8_t, 4>>();
    const sycl::vec<int8_t, 4> vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}
static inline int dp4a_ib(int a, int b, int c) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_IB_dp4a_ss(c, a, b, false);
#else
    return dp4a_cpp(a, b, c);
#endif
}
static inline int dp4a_us_ib(uint32_t a, int b, int c) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_IB_dp4a_us(c, (int) a, b, false);
#else
    return dp4a_us_cpp(a, b, c);
#endif
}

template <int V>
double run(sycl::queue& q, int* out, const int* in, int n, int iters) {
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int a = in[id], b = in[(id + 1) % n], c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k = 0; k < 1024; ++k) {
                if constexpr (V == 0) { c0 = dp4a_cpp(a, b, c0); c1 = dp4a_cpp(b, a, c1); c2 = dp4a_cpp(a ^ k, b, c2); c3 = dp4a_cpp(a, b ^ k, c3); }
                else if constexpr (V == 1) { c0 = dp4a_ib(a, b, c0); c1 = dp4a_ib(b, a, c1); c2 = dp4a_ib(a ^ k, b, c2); c3 = dp4a_ib(a, b ^ k, c3); }
                else if constexpr (V == 2) { c0 = dp4a_us_cpp((uint32_t) a, b, c0); c1 = dp4a_us_cpp((uint32_t) b, a, c1); c2 = dp4a_us_cpp((uint32_t) (a ^ k), b, c2); c3 = dp4a_us_cpp((uint32_t) a, b ^ k, c3); }
                else { c0 = dp4a_us_ib((uint32_t) a, b, c0); c1 = dp4a_us_ib((uint32_t) b, a, c1); c2 = dp4a_us_ib((uint32_t) (a ^ k), b, c2); c3 = dp4a_us_ib((uint32_t) a, b ^ k, c3); }
            }
            out[id] = c0 + c1 + c2 + c3;
        });
    q.wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const int n = 1 << 20;
    int* in = sycl::malloc_device<int>(n, q);
    int* o[4];
    for (auto& p : o) p = sycl::malloc_device<int>(n, q);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { in[i] = (int) (i * 2654435761u); }).wait();
    double t[4];
    run<0>(q, o[0], in, n, 1); t[0] = run<0>(q, o[0], in, n, 5);
    run<1>(q, o[1], in, n, 1); t[1] = run<1>(q, o[1], in, n, 5);
    run<2>(q, o[2], in, n, 1); t[2] = run<2>(q, o[2], in, n, 5);
    run<3>(q, o[3], in, n, 1); t[3] = run<3>(q, o[3], in, n, 5);
    std::vector<int> h[4];
    for (int v = 0; v < 4; ++v) { h[v].resize(n); q.memcpy(h[v].data(), o[v], n * 4).wait(); }
    int bad_ss = 0, bad_us = 0;
    for (int i = 0; i < n; ++i) { bad_ss += h[0][i] != h[1][i]; bad_us += h[2][i] != h[3][i]; }
    const double gops = (double) n * 4096 / 1e9;
    std::printf("ss cpp %.3f ms (%.0f G dp4a/s)  ss builtin %.3f ms (%.0f)  mismatches %d\n", t[0], gops / t[0] * 1e3, t[1], gops / t[1] * 1e3, bad_ss);
    std::printf("us cpp %.3f ms (%.0f G dp4a/s)  us builtin %.3f ms (%.0f)  mismatches %d\n", t[2], gops / t[2] * 1e3, t[3], gops / t[3] * 1e3, bad_us);
    return 0;
}
