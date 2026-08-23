// probe_graph_flags.cpp
// 1. Print device record/replay graph capability flags (non-zero = supported).
// 2. Wide name search for the instantiate + append entry points (experimental
//    API uses "Exp" suffix; instantiate may be under a non-obvious name).

#include <ze_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define RR_GRAPH_PROPS_STYPE 0x00020047u
typedef struct {
    ze_structure_type_t stype;
    void* pNext;
    uint32_t graphFlags;
} rr_graph_props_t;

int main() {
    zeInit(0);
    uint32_t nd = 0;
    zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drv(nd);
    zeDriverGet(&nd, drv.data());
    ze_driver_handle_t d = drv[0];

    uint32_t ndev = 0;
    zeDeviceGet(d, &ndev, nullptr);
    std::vector<ze_device_handle_t> dev(ndev);
    zeDeviceGet(d, &ndev, dev.data());
    ze_device_handle_t hdev = dev[0];

    // ---- device graph flags ----
    rr_graph_props_t gprops = {};
    gprops.stype = (ze_structure_type_t)RR_GRAPH_PROPS_STYPE;
    ze_device_properties_t dp = {};
    dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    dp.pNext = &gprops;
    zeDeviceGetProperties(hdev, &dp);
    printf("device: %s\n", dp.name);
    printf("=== device graph flags ===\n");
    printf("  graphFlags = 0x%x\n", gprops.graphFlags);
    printf("  IMMUTABLE_GRAPH    : %s\n", (gprops.graphFlags & 1u) ? "yes" : "no");
    printf("  MUTABLE_GRAPH      : %s\n", (gprops.graphFlags & 2u) ? "yes" : "no");
    printf("  SUBGRAPHS          : %s\n", (gprops.graphFlags & 4u) ? "yes" : "no");
    printf("  APPEND_COMMANDLIST : %s\n", (gprops.graphFlags & 8u) ? "yes" : "no");

    // ---- wide name search ----
    printf("=== wide name search ===\n");
    std::vector<std::string> candidates;
    const char* bases[] = {
        "zeGraphInstantiate", "zeGraphCreateExecutable", "zeExecutableGraphCreate",
        "zeGraphBuild", "zeGraphCompile", "zeGraphFinalize", "zeGraphPrepare",
        "zeGraphCreateInstance", "zeGraphMakeExecutable", "zeGraphUpload",
        "zeGraphDownload", "zeGraphLoad", "zeGraphStore",
    };
    const char* suffixes[] = { "", "Ext", "Exp" };
    for (auto* b : bases)
        for (auto* s : suffixes)
            candidates.push_back(std::string(b) + s);
    // also the append + a few others with all suffixes
    const char* more[] = {
        "zeCommandListAppendGraph", "zeGraphGetExecutable", "zeGraphToExecutable",
        "zeGraphConvert", "zeGraphWrap", "zeGraphBind",
    };
    for (auto* b : more)
        for (auto* s : suffixes)
            candidates.push_back(std::string(b) + s);

    int found = 0;
    for (auto& name : candidates) {
        void* fn = nullptr;
        zeDriverGetExtensionFunctionAddress(d, name.c_str(), &fn);
        if (fn) { printf("  FOUND: %s\n", name.c_str()); found++; }
    }
    printf("extra found: %d\n", found);
    return 0;
}