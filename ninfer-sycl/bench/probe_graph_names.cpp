// probe_graph_names.cpp
// Find the real function names for the record/replay graph extension.
// The driver advertises ZE_experimental_record_replay_graph (experimental),
// so functions likely use an "Exp" suffix (like zeCommandListCreateCloneExp),
// not the stable "Ext" suffix. Try a matrix of candidates.

#include <ze_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main() {
    zeInit(0);
    uint32_t nd = 0;
    zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drv(nd);
    zeDriverGet(&nd, drv.data());
    ze_driver_handle_t d = drv[0];

    // base names (stable "Ext" form)
    const char* bases[] = {
        "zeGraphCreate",
        "zeGraphDestroy",
        "zeGraphInstantiate",
        "zeGraphGetId",
        "zeGraphIsEmpty",
        "zeGraphDumpContents",
        "zeGraphSetDestructionCallback",
        "zeGraphPauseCapture",
        "zeGraphResumeCapture",
        "zeGraphGetPrimaryCommandList",
        "zeExecutableGraphDestroy",
        "zeExecutableGraphGetSourceGraph",
        "zeCommandListBeginCaptureIntoGraph",
        "zeCommandListBeginGraphCapture",
        "zeCommandListEndGraphCapture",
        "zeCommandListAppendGraph",
        "zeCommandListGetGraph",
        "zeCommandListIsGraphCaptureEnabled",
    };
    const char* suffixes[] = { "", "Ext", "Exp" };

    printf("=== name resolution matrix ===\n");
    int found = 0;
    for (auto* base : bases) {
        for (auto* suf : suffixes) {
            std::string name = std::string(base) + suf;
            void* fn = nullptr;
            zeDriverGetExtensionFunctionAddress(d, name.c_str(), &fn);
            if (fn) {
                printf("  FOUND: %s\n", name.c_str());
                found++;
            }
        }
    }
    printf("total found: %d\n", found);
    return 0;
}