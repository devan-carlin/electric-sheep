// Week-0 microbench 1: oneDNN GEMM on the Arc (SYCL/Level Zero backend).
//
// Goal: measure achieved TOPS for integer (u8s8 / s8s8) GEMM at the real
// Qwen3.8-27B prefill/decode shapes, and compare against a bf16 reference.
// A large gap between int8 and bf16 is the signal that XMX is being used;
// if int8 is at or below bf16, we are falling back to ALU math.
//
// Uses the oneDNN SYCL interop path (the GPU backend is built into libdnnl.so
// and driven through a SYCL device/context/queue).

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dnnl;
using namespace dnnl::sycl_interop;

struct Shape {
    const char* name;
    int N;  // output rows
    int K;  // input cols
};

// The real Qwen3.8-27B matrix shapes ([N,K] = out rows x in cols).
static const Shape kShapes[] = {
    {"mlp/gate_up", 34816, 5120},
    {"mlp/down",    5120, 17408},
    {"attn/query_key", 7168, 5120},
    {"attn/output",    5120, 6144},
    {"mtp/input_proj", 5120, 10240},
};

static const char* dt_name(memory::data_type dt) {
    switch (dt) {
        case memory::data_type::s8:   return "s8";
        case memory::data_type::u8:   return "u8";
        case memory::data_type::s32:  return "s32";
        case memory::data_type::bf16: return "bf16";
        case memory::data_type::f32:  return "f32";
        default: return "?";
    }
}

// Run one GEMM (M x K) @ (N x K) -> (M x N) and report ms + TOPS.
// Returns true if the primitive descriptor could be created (i.e. supported).
static bool run_gemm(sycl::queue& q, engine& eng, stream& st,
                     memory::data_type src_dt, memory::data_type w_dt,
                     memory::data_type dst_dt, int M, const Shape& sh,
                     int iters, const char* tag) {
    // oneDNN matmul: dst[M,N] = src[M,K] @ weights[K,N]
    memory::dims src_dims  = {M, sh.K};
    memory::dims w_dims    = {sh.K, sh.N};
    memory::dims dst_dims  = {M, sh.N};

    memory::desc src_md(src_dims, src_dt, memory::format_tag::any);
    memory::desc w_md(w_dims, w_dt, memory::format_tag::any);
    memory::desc dst_md(dst_dims, dst_dt, memory::format_tag::any);

    matmul::primitive_desc pd;
    try {
        pd = matmul::primitive_desc(eng, src_md, w_md, dst_md);
    } catch (const error& e) {
        printf("  %-22s M=%-6d %-4s x %-4s -> %-4s : NOT SUPPORTED (%s)\n",
               sh.name, M, dt_name(src_dt), dt_name(w_dt), dt_name(dst_dt),
               e.what());
        return false;
    }
    primitive prim(pd);

    // oneDNN may have picked blocked/padded formats; allocate to match.
    auto src_q = pd.query_md(query::exec_arg_md, DNNL_ARG_SRC);
    auto w_q   = pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS);
    auto dst_q = pd.query_md(query::exec_arg_md, DNNL_ARG_DST);

    size_t src_bytes = src_q.get_size();
    size_t w_bytes   = w_q.get_size();
    size_t dst_bytes = dst_q.get_size();

    auto* src_p = sycl::malloc_device<uint8_t>(src_bytes, q);
    auto* w_p   = sycl::malloc_device<uint8_t>(w_bytes, q);
    auto* dst_p = sycl::malloc_device<uint8_t>(dst_bytes, q);
    // NOTE: no h.fill here -- large fills (>=~100MB) trigger UR_DEVICE_LOST on
    // this driver. Data content is irrelevant for a TOPS benchmark; the warmup
    // loop below is what lets oneDNN JIT / pick an impl.

    memory src_mem = make_memory(src_q, eng, memory_kind::usm, src_p);
    memory w_mem   = make_memory(w_q, eng, memory_kind::usm, w_p);
    memory dst_mem = make_memory(dst_q, eng, memory_kind::usm, dst_p);

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, w_mem},
        {DNNL_ARG_DST, dst_mem},
    };

    // Warmup (also lets oneDNN JIT / pick impl).
    for (int i = 0; i < 3; i++) execute(prim, st, args);
    q.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) execute(prim, st, args);
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double ops = 2.0 * M * sh.N * sh.K;
    double tops = ops / (ms * 1e-3) / 1e12;

    printf("  %-22s M=%-6d %-4s x %-4s -> %-4s : %9.3f ms  %8.2f TOPS\n",
           sh.name, M, dt_name(src_dt), dt_name(w_dt), dt_name(dst_dt), ms, tops);

    sycl::free(src_p, q);
    sycl::free(w_p, q);
    sycl::free(dst_p, q);
    return true;
}

int main() {
    sycl::device dev{sycl::gpu_selector_v};
    sycl::queue q{dev};
    std::string name = dev.get_info<sycl::info::device::name>();
    auto ver = dev.get_info<sycl::info::device::driver_version>();
    printf("=== oneDNN GEMM microbench (SYCL/Level Zero) ===\n");
    printf("Device: %s\n", name.c_str());
    printf("Driver: %s\n", ver.c_str());
    printf("oneDNN: %d.%d.%d\n\n",
           dnnl::version()->major, dnnl::version()->minor,
           dnnl::version()->patch);

    engine eng = make_engine(dev, q.get_context());
    stream st  = make_stream(eng, q);

    const int M_PREFILL = 8192;
    const int M_DECODE  = 1;
    const int ITERS     = 50;

    printf("-- Integer GEMM (the XMX path) --\n");
    for (auto& sh : kShapes) {
        run_gemm(q, eng, st, memory::data_type::s8, memory::data_type::s8,
                 memory::data_type::s32, M_PREFILL, sh, ITERS, "prefill");
        run_gemm(q, eng, st, memory::data_type::s8, memory::data_type::s8,
                 memory::data_type::s32, M_DECODE, sh, ITERS, "decode");
    }

    printf("\n-- bf16 reference (ALU path, no XMX) --\n");
    for (auto& sh : kShapes) {
        run_gemm(q, eng, st, memory::data_type::bf16, memory::data_type::bf16,
                 memory::data_type::bf16, M_PREFILL, sh, ITERS, "prefill");
    }

    printf("\nDone.\n");
    return 0;
}