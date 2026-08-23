// Standalone diagnostic: dequant a few output_head rows + a token_embedding row,
// print their rms. Tells us whether the W8G32 weights are sane in magnitude.
#include "artifact/reader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace nsycl::artifact;

// f16 -> f32 (scales are stored as half-precision).
static float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = h & 0x03FFu;
    std::uint32_t f32;
    if (exp == 0) {
        if (mant == 0) f32 = sign;
        else { float f = static_cast<float>(mant) * 0x1p-15f; std::memcpy(&f32, &f, 4); f32 |= sign; }
    } else if (exp == 0x1F) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f32, 4);
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact>\n", argv[0]); return 1; }
    Reader store{std::filesystem::path(argv[1])};

    auto probe = [&](const std::string& name, int nrows) {
        const auto* obj = store.find(name);
        if (!obj) { std::fprintf(stderr, "missing %s\n", name.c_str()); return; }
        const auto& t = std::get<TensorDescriptor>(*obj);
        const int N = (int)t.shape[0];
        const int K = (int)t.shape[1];
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        auto span = store.payload(*obj);
        const std::uint8_t* base = reinterpret_cast<const std::uint8_t*>(span.data.data());
        const std::uint8_t* low   = base;
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(base + geo.scale_plane_offset);
        std::fprintf(stderr, "=== %s fmt=%d shape=[%d,%d] gpr=%d ===\n", name.c_str(),
                     (int)t.format, N, K, gpr);
        for (int n = 0; n < nrows && n < N; ++n) {
            const std::uint8_t* row_low   = low + (std::size_t)n * gpr * 32;
            const std::uint16_t* row_scale = scale + (std::size_t)n * gpr;
            double ss = 0.0; double mn = 1e30, mx = -1e30;
            double s0 = f16_to_f32(row_scale[0]);
            for (int g = 0; g < gpr; ++g) {
                double s = f16_to_f32(row_scale[g]);
                const std::uint8_t* codes = row_low + g * 32;
                for (int p = 0; p < 32; ++p) {
                    int qv = (int)(std::int8_t)codes[p];
                    double w = (double)qv * s;
                    ss += w * w; mn = std::min(mn, w); mx = std::max(mx, w);
                }
            }
            std::fprintf(stderr, "  row %d: rms=%.5f min=%.4f max=%.4f scale[0]=%.5f\n",
                         n, std::sqrt(ss / K), mn, mx, s0);
        }
    };

    probe("text/output_head", 4);
    probe("text/token_embedding", 4);

    // Probe specific rows: the observed top-5 + the expected "We" (1596).
    auto probe_row = [&](const std::string& name, int n) {
        const auto* obj = store.find(name);
        if (!obj) { std::fprintf(stderr, "missing %s\n", name.c_str()); return; }
        const auto& t = std::get<TensorDescriptor>(*obj);
        const int K = (int)t.shape[1];
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        auto span = store.payload(*obj);
        const std::uint8_t* base = reinterpret_cast<const std::uint8_t*>(span.data.data());
        const std::uint8_t* low   = base;
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(base + geo.scale_plane_offset);
        const std::uint8_t* row_low   = low + (std::size_t)n * gpr * 32;
        const std::uint16_t* row_scale = scale + (std::size_t)n * gpr;
        double ss = 0.0; double mn = 1e30, mx = -1e30;
        for (int g = 0; g < gpr; ++g) {
            double s = f16_to_f32(row_scale[g]);
            const std::uint8_t* codes = row_low + g * 32;
            for (int p = 0; p < 32; ++p) {
                int qv = (int)(std::int8_t)codes[p];
                double w = (double)qv * s;
                ss += w * w; mn = std::min(mn, w); mx = std::max(mx, w);
            }
        }
        std::fprintf(stderr, "  %s row %d: rms=%.5f min=%.4f max=%.4f\n", name.c_str(), n,
                     std::sqrt(ss / K), mn, mx);
    };
    std::fprintf(stderr, "=== raw scale+code dump (embedding) ===\n");
    {
        const auto* obj = store.find("text/token_embedding");
        const auto& t = std::get<TensorDescriptor>(*obj);
        const int K = (int)t.shape[1];
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        auto span = store.payload(*obj);
        const std::uint8_t* base = reinterpret_cast<const std::uint8_t*>(span.data.data());
        const std::uint8_t* low   = base;
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(base + geo.scale_plane_offset);
        auto dump = [&](int row, const char* tag) {
            const std::uint8_t* row_low   = low + (std::size_t)row * gpr * 32;
            const std::uint16_t* row_scale = scale + (std::size_t)row * gpr;
            std::fprintf(stderr, "  %s row %d: scales[0:8]=", tag, row);
            for (int g = 0; g < 8; ++g) std::fprintf(stderr, "%.5f ", f16_to_f32(row_scale[g]));
            std::fprintf(stderr, "\n    codes[0:16]=");
            for (int p = 0; p < 16; ++p) std::fprintf(stderr, "%d ", (int)(std::int8_t)row_low[p]);
            std::fprintf(stderr, "\n");
        };
        dump(197, "normal");
        dump(198, "INFLATED");
        dump(199, "normal");
        dump(10,  "normal");
        dump(11,  "INFLATED");
        dump(12,  "normal");
    }
    return 0;
}