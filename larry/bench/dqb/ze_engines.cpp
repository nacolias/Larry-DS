// ze_engines.cpp - larry: each Level Zero device's command queue groups (compute / copy, how many engines each) - is
// the B70's second copy engine (xe lists bcs0 + bcs8) open to user space, and as which ordinal?
//   build: icpx -O2 ze_engines.cpp -lze_loader -o ze_engines
#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>

int main() {
    if (zeInit(0) != ZE_RESULT_SUCCESS) { std::printf("zeInit failed\n"); return 1; }
    uint32_t nd = 0;
    zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drv(nd);
    zeDriverGet(&nd, drv.data());
    for (auto d : drv) {
        uint32_t ndev = 0;
        zeDeviceGet(d, &ndev, nullptr);
        std::vector<ze_device_handle_t> dev(ndev);
        zeDeviceGet(d, &ndev, dev.data());
        for (uint32_t i = 0; i < ndev; ++i) {
            ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(dev[i], &p);
            ze_pci_ext_properties_t pci = {ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
            zeDevicePciGetPropertiesExt(dev[i], &pci);
            uint32_t ng = 0;
            zeDeviceGetCommandQueueGroupProperties(dev[i], &ng, nullptr);
            std::vector<ze_command_queue_group_properties_t> g(ng, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
            zeDeviceGetCommandQueueGroupProperties(dev[i], &ng, g.data());
            std::printf("device %u %s (pci %02x:%02x.%x):\n", i, p.name, pci.address.bus, pci.address.device, pci.address.function);
            for (uint32_t k = 0; k < ng; ++k)
                std::printf("  ordinal %u: %s%s%s, %u engine(s), max fill pattern %zu\n", k,
                            (g[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ? "compute " : "",
                            (g[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ? "copy " : "",
                            (g[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS) ? "coop" : "",
                            g[k].numQueues, (size_t) g[k].maxMemoryFillPatternSize);
        }
    }
    return 0;
}
