// Diagnostic: does this Arc driver support the Level Zero command-list-clone
// (graph) extension? Tries both the directly-linked symbol and the
// zeDriverGetExtensionFunctionAddress path, and dumps driver extension props.
#include <level_zero/ze_api.h>
#include <level_zero/loader/ze_loader.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { ze_result_t _r = (x); if (_r != ZE_RESULT_SUCCESS) { \
    printf("  ZE error 0x%x at %s\n", (unsigned)_r, #x); return 1; } } while (0)

int main() {
    uint32_t ndrv = 0;
    CHECK(zeInit(0));
    CHECK(zeDriverGet(&ndrv, nullptr));
    std::vector<ze_driver_handle_t> drv(ndrv);
    CHECK(zeDriverGet(&ndrv, drv.data()));
    ze_driver_handle_t d = drv[0];

    // --- driver extension properties ---
    printf("=== Driver extension properties ===\n");
    uint32_t n = 0;
    ze_result_t r = zeDriverGetExtensionProperties(d, &n, nullptr);
    printf("  count rc=0x%x n=%u\n", (unsigned)r, n);
    if (r == ZE_RESULT_SUCCESS && n > 0) {
        std::vector<ze_driver_extension_properties_t> ext(n);
        uint32_t n2 = n;
        ze_result_t r2 = zeDriverGetExtensionProperties(d, &n2, ext.data());
        printf("  fetch rc=0x%x n2=%u\n", (unsigned)r2, n2);
        for (uint32_t i = 0; i < n2; i++)
            printf("  [%u] %s (api %u.%u)\n", i, ext[i].name,
                   ZE_MAJOR_VERSION(ext[i].version),
                   ZE_MINOR_VERSION(ext[i].version));
    }

    uint32_t ndev = 0;
    CHECK(zeDeviceGet(d, &ndev, nullptr));
    std::vector<ze_device_handle_t> dev(ndev);
    CHECK(zeDeviceGet(d, &ndev, dev.data()));
    ze_device_handle_t hdev = dev[0];

    // --- resolve the clone fn via the extension API ---
    printf("\n=== Resolve zeCommandListCreateCloneExp via extension API ===\n");
    void *fn = nullptr;
    r = zeDriverGetExtensionFunctionAddress(d, "zeCommandListCreateCloneExp", &fn);
    printf("  rc=0x%x fn=%p\n", (unsigned)r, fn);

    // --- try a real clone ---
    ze_context_desc_t cdesc{}; cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx; CHECK(zeContextCreate(d, &cdesc, &ctx));

    uint32_t nqg = 0;
    CHECK(zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, nullptr));
    std::vector<ze_command_queue_group_properties_t> qg(nqg);
    for (uint32_t i = 0; i < nqg; i++) {
        qg[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, qg.data() + i);
    }
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < nqg; i++)
        if (qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ordinal = i; break; }

    ze_command_list_desc_t cldesc{};
    cldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cldesc.commandQueueGroupOrdinal = ordinal;
    cldesc.flags = ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE;
    ze_command_list_handle_t cl;
    r = zeCommandListCreate(ctx, hdev, &cldesc, &cl);
    printf("\n  zeCommandListCreate(cloneable) rc=0x%x\n", (unsigned)r);
    if (r != ZE_RESULT_SUCCESS) { printf("  cannot create cloneable list\n"); return 0; }
    CHECK(zeCommandListClose(cl));

    ze_command_list_handle_t graph = nullptr;
    r = zeCommandListCreateCloneExp(cl, &graph);
    printf("  direct-linked clone rc=0x%x\n", (unsigned)r);

    if (fn) {
        auto clone_fn = (ze_result_t(*)(ze_command_list_handle_t, ze_command_list_handle_t*))fn;
        ze_command_list_handle_t g2 = nullptr;
        ze_result_t r2 = clone_fn(cl, &g2);
        printf("  extension-resolved clone rc=0x%x\n", (unsigned)r2);
    }

    printf("\nDone.\n");
    return 0;
}