#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
    sycl::device d(sycl::gpu_selector_v);
    std::printf("local_mem %zu, max wg %zu, CUs %u, sub-group sizes:", d.get_info<sycl::info::device::local_mem_size>(),
                d.get_info<sycl::info::device::max_work_group_size>(), d.get_info<sycl::info::device::max_compute_units>());
    for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", s);
    std::printf("\n");
}
