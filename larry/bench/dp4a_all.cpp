#include <sycl/sycl.hpp>
#include <cstdio>
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat) __attribute__((const));
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_su(int c, int a, int b, bool sat) __attribute__((const));
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_us(int c, int a, int b, bool sat) __attribute__((const));
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_uu(int c, int a, int b, bool sat) __attribute__((const));
#endif
int main() {
    sycl::queue q;
    int* o = sycl::malloc_shared<int>(8, q);
    q.single_task([=] {
        const int a = (int) 0x80FF017Fu, b = (int) 0x7F80FF01u;
#ifdef __SYCL_DEVICE_ONLY__
        o[0] = __builtin_IB_dp4a_ss(0, a, b, false); o[1] = __builtin_IB_dp4a_su(0, a, b, false);
        o[2] = __builtin_IB_dp4a_us(0, a, b, false); o[3] = __builtin_IB_dp4a_uu(0, a, b, false);
#else
        o[0] = a + b;
#endif
    }).wait();
    // CPU references
    auto ref = [](int a, int b, bool sa, bool sb) {
        int s = 0;
        for (int i = 0; i < 4; ++i) {
            const int x = sa ? (int) (int8_t) (a >> 8 * i) : (int) (uint8_t) (a >> 8 * i);
            const int y = sb ? (int) (int8_t) (b >> 8 * i) : (int) (uint8_t) (b >> 8 * i);
            s += x * y;
        }
        return s;
    };
    const int a = (int) 0x80FF017Fu, b = (int) 0x7F80FF01u;
    std::printf("ss %d/%d su %d/%d us %d/%d uu %d/%d\n", o[0], ref(a, b, 1, 1), o[1], ref(a, b, 1, 0), o[2], ref(a, b, 0, 1), o[3], ref(a, b, 0, 0));
}
