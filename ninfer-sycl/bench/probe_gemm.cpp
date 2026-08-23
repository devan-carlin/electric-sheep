// probe_gemm.cpp
// Minimal oneDNN SYCL-interop matmul probe. Isolates whether the GPU backend
// executes a matmul at all, and narrows DEVICE_LOST to a specific step.
// Enable verbose with -v / VERBOSE=1.

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

using namespace dnnl;
using namespace dnnl::sycl_interop;

static bool g_verbose = false;
#define VLOG(...) do { if (g_verbose) { printf(__VA_ARGS__); } } while (0)

// dst[M,N] = src[M,K] @ weights[K,N]
static bool run(sycl::queue& q, engine& eng, stream& st,
                memory::data_type dt, int M, int N, int K, int iters,
                bool do_fill = true) {
    memory::dims src_dims  = {M, K};
    memory::dims w_dims    = {K, N};
    memory::dims dst_dims  = {M, N};

    memory::desc src_md(src_dims, dt, memory::format_tag::any);
    memory::desc w_md(w_dims, dt, memory::format_tag::any);
    memory::desc dst_md(dst_dims, dt, memory::format_tag::any);

    matmul::primitive_desc pd;
    try {
        pd = matmul::primitive_desc(eng, src_md, w_md, dst_md);
    } catch (const error& e) {
        printf("  pd create FAILED: %s\n", e.what());
        return false;
    }
    VLOG("  pd created OK\n");
    primitive prim(pd);
    VLOG("  prim created OK\n");

    auto src_q = pd.query_md(query::exec_arg_md, DNNL_ARG_SRC);
    auto w_q   = pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS);
    auto dst_q = pd.query_md(query::exec_arg_md, DNNL_ARG_DST);
    size_t sb = src_q.get_size(), wb = w_q.get_size(), db = dst_q.get_size();
    VLOG("  sizes: src=%zu w=%zu dst=%zu\n", sb, wb, db);

    auto* sp = sycl::malloc_device<uint8_t>(sb, q);
    auto* wp = sycl::malloc_device<uint8_t>(wb, q);
    auto* dp = sycl::malloc_device<uint8_t>(db, q);
    VLOG("  malloc OK\n");

    if (do_fill) {
        q.submit([&](sycl::handler& h){ h.fill(sp, 1, sb); });
        q.submit([&](sycl::handler& h){ h.fill(wp, 1, wb); });
        q.wait();
        VLOG("  fills OK\n");
    } else {
        VLOG("  fills skipped\n");
    }

    memory sm = make_memory(src_q, eng, memory_kind::usm, sp);
    memory wm = make_memory(w_q,   eng, memory_kind::usm, wp);
    memory dm = make_memory(dst_q, eng, memory_kind::usm, dp);
    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, sm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, dm} };

    try {
        execute(prim, st, args);
        q.wait();
        VLOG("  first execute OK\n");
    } catch (const sycl::exception& e) {
        printf("  execute FAILED (sycl): %s\n", e.what());
        return false;
    } catch (const error& e) {
        printf("  execute FAILED (dnnl): %s\n", e.what());
        return false;
    }

    for (int i = 0; i < iters; i++) execute(prim, st, args);
    q.wait();
    VLOG("  %d iters OK\n", iters);

    sycl::free(sp, q); sycl::free(wp, q); sycl::free(dp, q);
    return true;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") g_verbose = true;
    if (const char* e = std::getenv("VERBOSE"); e && e[0]=='1') g_verbose = true;
    VLOG("verbose: %s\n", g_verbose ? "ON" : "off");

    sycl::device dev{sycl::gpu_selector_v};
    sycl::queue q{dev};
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    engine eng = make_engine(dev, q.get_context());
    stream st  = make_stream(eng, q);
    VLOG("engine/stream OK\n");

    // tiny bf16 first (most likely supported), then tiny int8
    printf("bf16 64x64x64   : ");
    bool a = run(q, eng, st, memory::data_type::bf16, 64, 64, 64, 5);
    printf("%s\n", a ? "OK" : "FAIL");

    printf("s8   64x64x64   : ");
    bool b = run(q, eng, st, memory::data_type::s8, 64, 64, 64, 5);
    printf("%s\n", b ? "OK" : "FAIL");

    // real mlp/gate_up shape: N=34816, K=5120, M=8192 (prefill)
    printf("bf16 8192x34816x5120 (no fill) : ");
    bool c = run(q, eng, st, memory::data_type::bf16, 8192, 34816, 5120, 3, false);
    printf("%s\n", c ? "OK" : "FAIL");

    printf("s8   8192x34816x5120 (no fill) : ");
    bool d = run(q, eng, st, memory::data_type::s8, 8192, 34816, 5120, 3, false);
    printf("%s\n", d ? "OK" : "FAIL");

    printf("Done.\n");
    return 0;
}