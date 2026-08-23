// Week-0 microbench 2: Level Zero graph capture round-trip.
//
// Goal: (a) confirm the graph path (zeCommandListCreateCloneExp on a
// ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE command list) is supported on this
// Arc driver, and (b) measure the host-side submission overhead of a
// pre-captured "graph" (cloned command list) vs. building+submitting a fresh
// command list each step. The delta is the launch overhead that CUDA-graphs /
// Level-Zero-graphs eliminate on the decode path.
//
// Work proxy: K device-to-device memory copies per "step" (no SPIR-V needed;
// we are measuring command-submission overhead, not compute).

#include <level_zero/ze_api.h>
#include <level_zero/loader/ze_loader.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(x) do { ze_result_t _r = (x); if (_r != ZE_RESULT_SUCCESS) { \
    printf("  ZE error %d at %s:%d (%s)\n", (int)_r, __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static const uint32_t K_COPIES = 64;      // copies per "step"
static const size_t   COPY_BYTES = 1u << 20; // 1 MiB each
static const int      STEPS = 2000;

static double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    printf("=== Level Zero graph capture microbench ===\n");

    // --- init loader ---
    uint32_t ndrv = 0;
    CHECK(zeInit(0));
    CHECK(zeDriverGet(&ndrv, nullptr));
    if (ndrv == 0) { printf("no drivers\n"); return 1; }
    std::vector<ze_driver_handle_t> drv(ndrv);
    CHECK(zeDriverGet(&ndrv, drv.data()));
    ze_driver_handle_t d = drv[0];

    uint32_t ndev = 0;
    CHECK(zeDeviceGet(d, &ndev, nullptr));
    std::vector<ze_device_handle_t> dev(ndev);
    CHECK(zeDeviceGet(d, &ndev, dev.data()));
    if (ndev == 0) { printf("no devices\n"); return 1; }
    ze_device_handle_t hdev = dev[0];

    ze_device_properties_t dp{};
    dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    CHECK(zeDeviceGetProperties(hdev, &dp));
    printf("Device: %s\n", dp.name);
    printf("  maxMemAllocSize: %.1f GiB\n", dp.maxMemAllocSize / 1073741824.0);

    // --- context / queue / fence ---
    ze_context_desc_t cdesc{};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx;
    CHECK(zeContextCreate(d, &cdesc, &ctx));

    // pick a compute-capable command queue group ordinal
    uint32_t nqg = 0;
    CHECK(zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, nullptr));
    std::vector<ze_command_queue_group_properties_t> qg(nqg);
    for (uint32_t i = 0; i < nqg; i++) {
        qg[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        CHECK(zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, qg.data() + i));
    }
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < nqg; i++)
        if (qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ordinal = i; break; }

    ze_command_queue_desc_t qdesc{};
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.ordinal = ordinal;
    qdesc.index = 0;
    qdesc.flags = 0;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t q;
    CHECK(zeCommandQueueCreate(ctx, hdev, &qdesc, &q));

    ze_fence_desc_t fdesc{};
    fdesc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
    ze_fence_handle_t fence;
    CHECK(zeFenceCreate(q, &fdesc, &fence));

    // --- device buffers ---
    ze_device_mem_alloc_desc_t mdesc{};
    mdesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    mdesc.ordinal = 0;
    void *src, *dst;
    CHECK(zeMemAllocDevice(ctx, &mdesc, COPY_BYTES, 256, hdev, &src));
    CHECK(zeMemAllocDevice(ctx, &mdesc, COPY_BYTES, 256, hdev, &dst));

    // --- build a cloneable command list with K copies ---
    ze_command_list_desc_t cldesc{};
    cldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cldesc.commandQueueGroupOrdinal = ordinal;
    cldesc.flags = ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE;
    ze_command_list_handle_t cl;
    CHECK(zeCommandListCreate(ctx, hdev, &cldesc, &cl));
    for (uint32_t i = 0; i < K_COPIES; i++)
        CHECK(zeCommandListAppendMemoryCopy(cl, dst, src, COPY_BYTES, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cl));

    // --- (a) is the clone (graph) path supported? ---
    ze_command_list_handle_t graph = nullptr;
    ze_result_t clone_rc = zeCommandListCreateCloneExp(cl, &graph);
    printf("\nGraph support (zeCommandListCreateCloneExp): %s\n",
           clone_rc == ZE_RESULT_SUCCESS ? "SUPPORTED" : "NOT SUPPORTED");
    if (clone_rc != ZE_RESULT_SUCCESS) {
        printf("  clone rc = %d\n", (int)clone_rc);
        printf("  -> CUDA-graphs analog unavailable; decode loses the launch win.\n");
        return 0;
    }

    // --- (b) host-side submission overhead: fresh list vs graph ---
    // Warmup.
    for (int i = 0; i < 50; i++) {
        CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, fence));
        CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    }

    // Non-graph: build a fresh command list each step, then submit.
    {
        double t0 = now_ns();
        for (int s = 0; s < STEPS; s++) {
            ze_command_list_handle_t f;
            CHECK(zeCommandListCreate(ctx, hdev, &cldesc, &f));
            for (uint32_t i = 0; i < K_COPIES; i++)
                CHECK(zeCommandListAppendMemoryCopy(f, dst, src, COPY_BYTES, nullptr, 0, nullptr));
            CHECK(zeCommandListClose(f));
            CHECK(zeCommandQueueExecuteCommandLists(q, 1, &f, fence));
            CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
            CHECK(zeCommandListDestroy(f));
        }
        double t1 = now_ns();
        printf("\nNon-graph (build+submit %u copies/step): %8.1f us/step\n",
               K_COPIES, (t1 - t0) / STEPS / 1000.0);
    }

    // Graph: submit the pre-cloned graph each step.
    {
        double t0 = now_ns();
        for (int s = 0; s < STEPS; s++) {
            CHECK(zeCommandQueueExecuteCommandLists(q, 1, &graph, fence));
            CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
        }
        double t1 = now_ns();
        printf("Graph     (replay %u copies/step):     %8.1f us/step\n",
               K_COPIES, (t1 - t0) / STEPS / 1000.0);
    }

    // Host-only submission cost (no fence wait) to isolate launch overhead.
    {
        double t0 = now_ns();
        for (int s = 0; s < STEPS; s++) {
            ze_command_list_handle_t f;
            CHECK(zeCommandListCreate(ctx, hdev, &cldesc, &f));
            for (uint32_t i = 0; i < K_COPIES; i++)
                CHECK(zeCommandListAppendMemoryCopy(f, dst, src, COPY_BYTES, nullptr, 0, nullptr));
            CHECK(zeCommandListClose(f));
            CHECK(zeCommandQueueExecuteCommandLists(q, 1, &f, nullptr));
            CHECK(zeCommandListDestroy(f));
        }
        CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));
        double t1 = now_ns();
        printf("\nHost-only submit, non-graph: %8.1f us/step\n", (t1 - t0) / STEPS / 1000.0);

        t0 = now_ns();
        for (int s = 0; s < STEPS; s++)
            CHECK(zeCommandQueueExecuteCommandLists(q, 1, &graph, nullptr));
        CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));
        t1 = now_ns();
        printf("Host-only submit, graph:     %8.1f us/step\n", (t1 - t0) / STEPS / 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}