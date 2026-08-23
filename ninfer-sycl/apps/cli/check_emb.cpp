// CPU check: dequantize embedding row 9707 from the artifact and print stats.
#include "artifact/reader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = h & 0x03FFu;
    std::uint32_t f32;
    if (exp == 0) {
        if (mant == 0) f32 = sign;
        else {
            float f = static_cast<float>(mant) * 0x1p-15f;
            std::memcpy(&f32, &f, 4);
            f32 |= sign;
        }
    } else if (exp == 0x1F) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f32, 4);
    return r;
}

int main() {
    nsycl::artifact::Reader reader("models/qwen3_8_27b.ninfer");
    const auto* t = reader.find("text/token_embedding");
    if (!t) { std::fprintf(stderr, "no embedding tensor\n"); return 1; }
    const auto* td = std::get_if<nsycl::artifact::TensorDescriptor>(t);
    std::printf("embedding: shape=[%llu,%llu] fmt=%s layout=%s bytes=%llu\n",
                (unsigned long long)td->shape[0], (unsigned long long)td->shape[1],
                std::string(nsycl::artifact::format_name(td->format)).c_str(),
                std::string(nsycl::artifact::layout_name(td->layout)).c_str(),
                (unsigned long long)td->bytes);

    const std::uint64_t shape[2] = {td->shape[0], td->shape[1]};
    auto geo = nsycl::artifact::row_split_geometry(td->format,
                                                   std::span<const std::uint64_t>(shape, 2));
    std::printf("geo: padded_cols=%llu groups_per_row=%llu low_bpg=%llu high_bpg=%llu\n",
                (unsigned long long)geo.padded_columns, (unsigned long long)geo.groups_per_row,
                (unsigned long long)geo.low_bytes_per_group, (unsigned long long)geo.high_bytes_per_group);
    std::printf("geo: low_plane=%llu high_off=%llu high_bytes=%llu scale_off=%llu scale_bytes=%llu encoded=%llu\n",
                (unsigned long long)geo.low_plane_bytes, (unsigned long long)geo.high_plane_offset,
                (unsigned long long)geo.high_plane_bytes, (unsigned long long)geo.scale_plane_offset,
                (unsigned long long)geo.scale_plane_bytes, (unsigned long long)geo.encoded_bytes);

    const int row = 9707;
    const int K = 5120;
    auto span = reader.payload(*t);
    const std::uint8_t* base = reinterpret_cast<const std::uint8_t*>(span.data.data());
    const std::uint8_t* low = base;
    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(base + geo.scale_plane_offset);

    const std::uint8_t* row_low = low + (std::size_t)row * geo.groups_per_row * 32;
    const std::uint16_t* row_scale = scale + (std::size_t)row * geo.groups_per_row;

    // Print first 8 scales.
    std::printf("scales[0:8] =");
    for (int g = 0; g < 8; ++g) std::printf(" %.6g", f16_to_f32(row_scale[g]));
    std::printf("\n");

    // Dequantize the row.
    std::vector<float> vals(K);
    double ss = 0.0;
    for (int k = 0; k < K; ++k) {
        int g = k / 32, p = k % 32;
        const std::uint8_t* codes = row_low + (std::size_t)g * 32;
        float s = f16_to_f32(row_scale[g]);
        int qv = static_cast<std::int8_t>(codes[p]);
        vals[k] = static_cast<float>(qv) * s;
        ss += (double)vals[k] * vals[k];
    }
    std::printf("row %d: rms=%.6f min=%.6f max=%.6f\n", row, std::sqrt(ss / K),
                *std::min_element(vals.begin(), vals.end()),
                *std::max_element(vals.begin(), vals.end()));
    std::printf("first 16 vals:");
    for (int k = 0; k < 16; ++k) std::printf(" %.5f", vals[k]);
    std::printf("\n");
    return 0;
}