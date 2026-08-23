// Validate the GEMV kernels against a CPU reference dequant on a real tensor.
//
// For a chosen quantized tensor, dequantize it fully on the CPU (using the same
// dequant math as the kernels), compute the reference dot product with a random
// bf16 activation, and compare against the SYCL GEMV output.

#include "artifact/reader.h"
#include "ops/linear/gemv.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace nsycl::artifact;

namespace {

// CPU dequant of one row of a row-split tensor into a float vector of length K.
std::vector<float> dequant_row_cpu(const TensorDescriptor& t, const std::byte* base, int n) {
    const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
    auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
    const int K = static_cast<int>(t.shape[1]);
    const int gpr = static_cast<int>(geo.groups_per_row);
    const std::uint8_t* low  = reinterpret_cast<const std::uint8_t*>(base);
    const std::uint8_t* high = low + geo.high_plane_offset;
    const std::uint16_t* scale =
        reinterpret_cast<const std::uint16_t*>(low + geo.scale_plane_offset);

    std::vector<float> w(K);
    const std::uint8_t* row_low   = low + static_cast<std::size_t>(n) * gpr * 32;
    const std::uint8_t* row_high  = high + static_cast<std::size_t>(n) * gpr * geo.high_bytes_per_group;
    const std::uint16_t* row_scale = scale + static_cast<std::size_t>(n) * gpr;

    for (int g = 0; g < gpr; ++g) {
        float s = nsycl::ops::f16_to_f32(row_scale[g]);
        const std::uint8_t* codes = row_low + g * 32;
        const std::uint8_t* hbits = row_high + g * geo.high_bytes_per_group;
        int per_group = (t.format == NumericFormat::W8G32_F16S) ? 32 : 64;
        for (int p = 0; p < per_group; ++p) {
            int k = g * per_group + p;
            if (k >= K) break;
            int qv;
            if (t.format == NumericFormat::Q4G64_F16S) {
                std::uint8_t byte = codes[p >> 1];
                int code = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                qv = (code ^ 8) - 8;
            } else if (t.format == NumericFormat::Q5G64_F16S) {
                std::uint8_t byte = codes[p >> 1];
                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                int hi1 = (hbits[p >> 3] >> (p & 7)) & 1;
                int u = lo4 | (hi1 << 4);
                qv = (u ^ 16) - 16;
            } else if (t.format == NumericFormat::Q6G64_F16S) {
                std::uint8_t byte = codes[p >> 1];
                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                int hi2 = (hbits[p >> 1] >> ((p & 1) * 4)) & 0x03;
                int u = lo4 | (hi2 << 4);
                qv = (u ^ 32) - 32;
            } else { // W8
                qv = static_cast<std::int8_t>(codes[p]);
            }
            w[k] = static_cast<float>(qv) * s;
        }
    }
    return w;
}

// Double-precision dot of a dequantized row with x (bf16). Isolates dequant
// correctness from f32 accumulation-order noise.
double dot_double(const std::vector<float>& w, const std::vector<std::uint16_t>& x) {
    double acc = 0.0;
    for (std::size_t k = 0; k < w.size(); ++k) {
        acc += static_cast<double>(w[k]) * static_cast<double>(nsycl::ops::bf16_to_f32(x[k]));
    }
    return acc;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact> [tensor_name]\n", argv[0]); return 2; }
    const std::string tensor_name = (argc >= 3) ? argv[2] : "text/layers/0/gdn/query_key";

    Reader reader(argv[1]);
    const auto* t = std::get_if<TensorDescriptor>(&reader.find(tensor_name)[0]);
    if (!t) { std::fprintf(stderr, "tensor %s not found\n", tensor_name.c_str()); return 1; }
    if (t->layout != StorageLayout::RowSplitK128V1) {
        std::fprintf(stderr, "tensor %s is not row-split\n", tensor_name.c_str()); return 1;
    }

    const int N = static_cast<int>(t->shape[0]);
    const int K = static_cast<int>(t->shape[1]);
    const std::uint64_t shape[2] = {t->shape[0], t->shape[1]};
    auto geo = row_split_geometry(t->format, std::span<const std::uint64_t>(shape, 2));
    const int gpr = static_cast<int>(geo.groups_per_row);

    std::printf("tensor=%s shape=[%d,%d] format=%s groups_per_row=%d\n", tensor_name.c_str(), N, K,
                std::string(format_name(t->format)).c_str(), gpr);

    const std::byte* base = reader.payload(*t).data.data();
    const std::uint8_t* low  = reinterpret_cast<const std::uint8_t*>(base);
    const std::uint8_t* high = low + geo.high_plane_offset;
    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(low + geo.scale_plane_offset);

    sycl::queue q;

    // Copy the 3 planes into shared USM (device-readable). The real engine does
    // the same at load time.
    const std::size_t low_bytes   = geo.low_plane_bytes;
    const std::size_t high_bytes  = geo.high_plane_bytes;
    const std::size_t scale_bytes = geo.scale_plane_bytes;
    auto* low_usm   = sycl::malloc_device<std::uint8_t>(low_bytes, q);
    auto* high_usm  = sycl::malloc_device<std::uint8_t>(high_bytes > 0 ? high_bytes : 1, q);
    auto* scale_usm = sycl::malloc_device<std::uint16_t>(scale_bytes / 2, q);
    q.memcpy(low_usm, low, low_bytes);
    if (high_bytes > 0) q.memcpy(high_usm, high, high_bytes);
    q.memcpy(scale_usm, scale, scale_bytes);
    q.wait();

    // Random bf16 activation in [-1, 1].
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::uint16_t> x(K);
    for (int k = 0; k < K; ++k) {
        float v = dist(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        // round-to-nearest-even to bf16
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        x[k] = static_cast<std::uint16_t>(bits >> 16);
    }

    auto* x_usm   = sycl::malloc_device<std::uint16_t>(K, q);
    auto* out_usm = sycl::malloc_device<float>(N, q);
    q.memcpy(x_usm, x.data(), K * sizeof(std::uint16_t));
    q.wait();

    switch (t->format) {
    case NumericFormat::Q4G64_F16S:
        nsycl::ops::gemv_q4(q, low_usm, scale_usm, x_usm, out_usm, N, K, gpr);
        break;
    case NumericFormat::Q5G64_F16S:
        nsycl::ops::gemv_q5(q, low_usm, high_usm, scale_usm, x_usm, out_usm, N, K, gpr);
        break;
    case NumericFormat::Q6G64_F16S:
        nsycl::ops::gemv_q6(q, low_usm, high_usm, scale_usm, x_usm, out_usm, N, K, gpr);
        break;
    case NumericFormat::W8G32_F16S:
        nsycl::ops::gemv_w8(q, low_usm, scale_usm, x_usm, out_usm, N, K, gpr);
        break;
    default:
        std::fprintf(stderr, "unsupported format\n");
        return 1;
    }
    q.wait();

    std::vector<float> out_host(N);
    q.memcpy(out_host.data(), out_usm, N * sizeof(float));
    q.wait();

    // CPU reference: dequant each row, dot with x in double precision. This
    // isolates dequant correctness from f32 accumulation-order noise.
    double max_rel = 0.0;
    int bad = 0;
    for (int n = 0; n < N; ++n) {
        std::vector<float> w = dequant_row_cpu(*t, base, n);
        double ref = dot_double(w, x);
        double diff = std::fabs(static_cast<double>(out_host[n]) - ref);
        double scale_ref = std::fmax(std::fabs(ref), 1e-6);
        double rel = diff / scale_ref;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3) ++bad;
    }

    std::printf("max_rel_err=%.3e  bad(>1e-3)=%d/%d\n", max_rel, bad, N);
    sycl::free(x_usm, q);
    sycl::free(out_usm, q);
    sycl::free(low_usm, q);
    sycl::free(high_usm, q);
    sycl::free(scale_usm, q);
    return bad == 0 ? 0 : 1;
}