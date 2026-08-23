// probe_graph.cpp  (experimental record/replay graph, "Exp" suffix)
//
// The Arc driver advertises ZE_experimental_record_replay_graph. Its functions
// use the "Exp" suffix and are resolved via zeDriverGetExtensionFunctionAddress.
// Signatures come from intel/compute-runtime
// level_zero/include/level_zero/driver_experimental/zex_graph.h:
//   zeGraphCreateExp(ctx, phGraph, pNext)                 [pNext is LAST]
//   zeCommandListBeginCaptureIntoGraphExp(cl, hGraph, pNext)
//   zeCommandListEndGraphCaptureExp(cl, phGraph, pNext)
//   zeCommandListInstantiateGraphExp(hGraph, phExecGraph, pNext)
//   zeCommandListAppendGraphExp(cl, hExecGraph, pNext, hSignal, nWait, pWait)
//
// Flow: GraphCreateExp -> BeginCaptureIntoGraphExp (immediate CL) -> append N
// copies -> EndGraphCaptureExp -> InstantiateGraphExp -> AppendGraphExp (fresh
// CL) -> execute -> verify. Then time host-side submission overhead: graph vs
// fresh command list.

#include <ze_api.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// ---- verbose logging (off by default; enable with -v/--verbose or VERBOSE=1) ----
static bool g_verbose = false;
#define VLOG(...) do { if (g_verbose) { printf(__VA_ARGS__); } } while (0)

// ---- graph handle types (absent from local 1.28.6 header) ----
typedef struct _ze_graph_handle_t *ze_graph_handle_t;
typedef struct _ze_executable_graph_handle_t *ze_executable_graph_handle_t;

// ---- experimental graph function-pointer types (from zex_graph.h) ----
typedef ze_result_t (ZE_APICALL *pfnGraphCreateExp_t)(
    ze_context_handle_t, ze_graph_handle_t*, void*);
typedef ze_result_t (ZE_APICALL *pfnBeginCaptureIntoGraphExp_t)(
    ze_command_list_handle_t, ze_graph_handle_t, void*);
typedef ze_result_t (ZE_APICALL *pfnEndGraphCaptureExp_t)(
    ze_command_list_handle_t, ze_graph_handle_t*, void*);
typedef ze_result_t (ZE_APICALL *pfnInstantiateGraphExp_t)(
    ze_graph_handle_t, ze_executable_graph_handle_t*, void*);
typedef ze_result_t (ZE_APICALL *pfnAppendGraphExp_t)(
    ze_command_list_handle_t, ze_executable_graph_handle_t, void*,
    ze_event_handle_t, uint32_t, ze_event_handle_t*);
typedef ze_result_t (ZE_APICALL *pfnGraphDestroyExp_t)(ze_graph_handle_t);
typedef ze_result_t (ZE_APICALL *pfnExecGraphDestroyExp_t)(ze_executable_graph_handle_t);

static const char* rcstr(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "SUCCESS";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "DEVICE_LOST";
        case 0x78000024: return "INVALID_GRAPH";
        case 0x78000025: return "GRAPH_CAPTURE_UNSUPPORTED";
        case ZE_RESULT_ERROR_UNKNOWN: return "UNKNOWN";
        default: return "other";
    }
}

#define CHECK(x) do { ze_result_t _r = (x); if (_r != ZE_RESULT_SUCCESS) { \
    printf("  ZE error 0x%x (%s) at %s:%d\n", (unsigned)_r, rcstr(_r), #x, __LINE__); \
    return 1; } } while (0)

int main(int argc, char** argv) {
    using clk = std::chrono::steady_clock;

    // verbose flag: -v / --verbose / VERBOSE=1
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") g_verbose = true;
    }
    if (const char* e = std::getenv("VERBOSE"); e && e[0] == '1') g_verbose = true;
    VLOG("verbose logging: %s\n", g_verbose ? "ON" : "off");

    ze_result_t r = zeInit(0);
    if (r != ZE_RESULT_SUCCESS) { printf("zeInit rc=0x%x\n", (unsigned)r); return 1; }

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

    ze_device_properties_t dp = {};
    dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(hdev, &dp);
    printf("device: %s\n", dp.name);

    // ---- resolve experimental graph fns ----
    pfnGraphCreateExp_t fCreate = nullptr;
    pfnBeginCaptureIntoGraphExp_t fBegin = nullptr;
    pfnEndGraphCaptureExp_t fEnd = nullptr;
    pfnInstantiateGraphExp_t fInst = nullptr;
    pfnAppendGraphExp_t fAppend = nullptr;
    pfnGraphDestroyExp_t fDestroy = nullptr;
    pfnExecGraphDestroyExp_t fExecDestroy = nullptr;

    zeDriverGetExtensionFunctionAddress(d, "zeGraphCreateExp", (void**)&fCreate);
    zeDriverGetExtensionFunctionAddress(d, "zeCommandListBeginCaptureIntoGraphExp", (void**)&fBegin);
    zeDriverGetExtensionFunctionAddress(d, "zeCommandListEndGraphCaptureExp", (void**)&fEnd);
    zeDriverGetExtensionFunctionAddress(d, "zeCommandListInstantiateGraphExp", (void**)&fInst);
    zeDriverGetExtensionFunctionAddress(d, "zeCommandListAppendGraphExp", (void**)&fAppend);
    zeDriverGetExtensionFunctionAddress(d, "zeGraphDestroyExp", (void**)&fDestroy);
    zeDriverGetExtensionFunctionAddress(d, "zeExecutableGraphDestroyExp", (void**)&fExecDestroy);

    printf("=== resolve experimental graph fns ===\n");
    printf("  zeGraphCreateExp                     : %s\n", fCreate ? "OK" : "nil");
    printf("  zeCommandListBeginCaptureIntoGraphExp: %s\n", fBegin ? "OK" : "nil");
    printf("  zeCommandListEndGraphCaptureExp      : %s\n", fEnd ? "OK" : "nil");
    printf("  zeCommandListInstantiateGraphExp     : %s\n", fInst ? "OK" : "nil");
    printf("  zeCommandListAppendGraphExp          : %s\n", fAppend ? "OK" : "nil");
    printf("  zeGraphDestroyExp                    : %s\n", fDestroy ? "OK" : "nil");
    printf("  zeExecutableGraphDestroyExp          : %s\n", fExecDestroy ? "OK" : "nil");
    if (!fCreate || !fBegin || !fEnd || !fInst || !fAppend) {
        printf("-> missing fns; stopping.\n");
        return 1;
    }

    // ---- context + queue ----
    VLOG("step: context\n");
    ze_context_desc_t cdesc = {};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx = nullptr;
    CHECK(zeContextCreate(d, &cdesc, &ctx));
    VLOG("step: queue groups\n");

    uint32_t nqg = 0;
    zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, nullptr);
    std::vector<ze_command_queue_group_properties_t> qg(nqg);
    for (uint32_t i = 0; i < nqg; i++) {
        qg[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        zeDeviceGetCommandQueueGroupProperties(hdev, &nqg, qg.data() + i);
    }
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < nqg; i++)
        if (qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ordinal = i; break; }

    ze_command_queue_desc_t qdesc = {};
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.ordinal = ordinal;
    qdesc.index = 0;
    qdesc.flags = 0;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t q = nullptr;
    CHECK(zeCommandQueueCreate(ctx, hdev, &qdesc, &q));
    VLOG("step: queue created (ordinal=%u)\n", ordinal);

    // ---- shared buffers for copies ----
    const size_t COPY_BYTES = 1u << 20;  // 1 MiB
    const uint32_t NCOPIES = 64;
    ze_device_mem_alloc_desc_t mdesc = {};
    mdesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    mdesc.ordinal = 0;
    ze_host_mem_alloc_desc_t hdesc = {};
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *src = nullptr, *dst = nullptr;
    VLOG("step: zeMemAllocShared src\n");
    CHECK(zeMemAllocShared(ctx, &mdesc, &hdesc, COPY_BYTES, 256, hdev, &src));
    VLOG("step: zeMemAllocShared dst\n");
    CHECK(zeMemAllocShared(ctx, &mdesc, &hdesc, COPY_BYTES, 256, hdev, &dst));
    VLOG("step: mem alloc (src=%p dst=%p)\n", (void*)src, (void*)dst);
    std::memset(src, 0xAB, COPY_BYTES);
    std::memset(dst, 0x00, COPY_BYTES);

    // ---- capture N copies into a graph ----
    printf("=== capture ===\n");
    VLOG("step: GraphCreateExp\n");
    ze_graph_handle_t graph = nullptr;
    ze_result_t rc = fCreate(ctx, &graph, nullptr);
    printf("  GraphCreateExp rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
    if (rc != ZE_RESULT_SUCCESS) return 1;

    VLOG("step: CreateImmediate\n");
    ze_command_list_handle_t immCl = nullptr;
    rc = zeCommandListCreateImmediate(ctx, hdev, &qdesc, &immCl);
    printf("  CreateImmediate rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
    if (rc != ZE_RESULT_SUCCESS) return 1;

    VLOG("step: BeginCapture\n");
    rc = fBegin(immCl, graph, nullptr);
    printf("  BeginCaptureIntoGraphExp rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
    if (rc != ZE_RESULT_SUCCESS) return 1;

    VLOG("step: append %d copies\n", NCOPIES);
    for (uint32_t i = 0; i < NCOPIES; i++)
        zeCommandListAppendMemoryCopy(immCl, dst, src, COPY_BYTES, nullptr, 0, nullptr);

    VLOG("step: EndCapture\n");
    ze_graph_handle_t captured = nullptr;
    rc = fEnd(immCl, &captured, nullptr);
    printf("  EndGraphCaptureExp rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
    if (rc != ZE_RESULT_SUCCESS) return 1;
    VLOG("step: captured=%p\n", (void*)captured);

    VLOG("step: InstantiateGraphExp\n");
    ze_executable_graph_handle_t execGraph = nullptr;
    rc = fInst(captured, &execGraph, nullptr);
    printf("  InstantiateGraphExp rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
    if (rc != ZE_RESULT_SUCCESS) return 1;
    VLOG("step: execGraph=%p\n", (void*)execGraph);

    // ---- execute the graph on an immediate CL (must match the capture CL's mode) ----
    {
        VLOG("step: execute (immediate)\n");
        ze_command_list_handle_t imm2 = nullptr;
        zeCommandListCreateImmediate(ctx, hdev, &qdesc, &imm2);
        VLOG("step: AppendGraphExp\n");
        rc = fAppend(imm2, execGraph, nullptr, nullptr, 0, nullptr);
        printf("  AppendGraphExp(immediate) rc=0x%x (%s)\n", (unsigned)rc, rcstr(rc));
        if (rc != ZE_RESULT_SUCCESS) return 1;
        VLOG("step: sync\n");
        zeCommandQueueSynchronize(q, UINT64_MAX);
        zeCommandListDestroy(imm2);
    }

    // ---- verify ----
    const uint8_t* dstBytes = (const uint8_t*)dst;
    bool ok = true;
    for (size_t i = 0; i < COPY_BYTES; i += 4096)
        if (dstBytes[i] != 0xAB) { ok = false; break; }
    printf("=== verify ===\n");
    printf("  dst filled with 0xAB: %s\n", ok ? "YES (graph executed correctly)" : "NO (graph did not run)");

    // ---- host-side submission overhead ----
    printf("=== host overhead (K=2000 iters, %d copies/step) ===\n", NCOPIES);
    const int K = 2000;

    // Path A: fresh non-immediate CL, N appends, close, execute
    {
        for (int w = 0; w < 50; w++) {
            ze_command_list_handle_t cl = nullptr;
            ze_command_list_desc_t clDesc = {};
            clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            clDesc.commandQueueGroupOrdinal = ordinal;
            zeCommandListCreate(ctx, hdev, &clDesc, &cl);
            for (uint32_t i = 0; i < NCOPIES; i++)
                zeCommandListAppendMemoryCopy(cl, dst, src, COPY_BYTES, nullptr, 0, nullptr);
            zeCommandListClose(cl);
            zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr);
            zeCommandListDestroy(cl);
        }
        zeCommandQueueSynchronize(q, UINT64_MAX);

        auto t0 = clk::now();
        for (int i = 0; i < K; i++) {
            ze_command_list_handle_t cl = nullptr;
            ze_command_list_desc_t clDesc = {};
            clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            clDesc.commandQueueGroupOrdinal = ordinal;
            zeCommandListCreate(ctx, hdev, &clDesc, &cl);
            for (uint32_t c = 0; c < NCOPIES; c++)
                zeCommandListAppendMemoryCopy(cl, dst, src, COPY_BYTES, nullptr, 0, nullptr);
            zeCommandListClose(cl);
            zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr);
            zeCommandListDestroy(cl);
        }
        auto t1 = clk::now();
        zeCommandQueueSynchronize(q, UINT64_MAX);
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 / K;
        printf("  A) fresh CL + %d appends : %.1f us/step (host)\n", NCOPIES, us);
    }

    // Path B: fresh immediate CL, 1 graph append (auto-executes), destroy
    {
        for (int w = 0; w < 50; w++) {
            ze_command_list_handle_t imm2 = nullptr;
            zeCommandListCreateImmediate(ctx, hdev, &qdesc, &imm2);
            fAppend(imm2, execGraph, nullptr, nullptr, 0, nullptr);
            zeCommandListDestroy(imm2);
        }
        zeCommandQueueSynchronize(q, UINT64_MAX);

        auto t0 = clk::now();
        for (int i = 0; i < K; i++) {
            ze_command_list_handle_t imm2 = nullptr;
            zeCommandListCreateImmediate(ctx, hdev, &qdesc, &imm2);
            fAppend(imm2, execGraph, nullptr, nullptr, 0, nullptr);
            zeCommandListDestroy(imm2);
        }
        auto t1 = clk::now();
        zeCommandQueueSynchronize(q, UINT64_MAX);
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 / K;
        printf("  B) fresh imm CL + 1 graph: %.1f us/step (host)\n", us);
    }

    // ---- cleanup ----
    VLOG("cleanup: execGraph\n");
    if (fExecDestroy) fExecDestroy(execGraph);
    VLOG("cleanup: captured\n");
    if (fDestroy) fDestroy(captured);
    VLOG("cleanup: graph (same as captured? %s)\n", graph == captured ? "yes" : "no");
    if (fDestroy && graph != captured) fDestroy(graph);
    VLOG("cleanup: immCl\n");
    zeCommandListDestroy(immCl);
    VLOG("cleanup: mem\n");
    zeMemFree(ctx, src);
    zeMemFree(ctx, dst);
    VLOG("cleanup: queue\n");
    zeCommandQueueDestroy(q);
    VLOG("cleanup: context\n");
    zeContextDestroy(ctx);
    VLOG("cleanup: done\n");

    printf("Done.\n");
    return 0;
}