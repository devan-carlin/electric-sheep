#pragma once

// nsycl::ops - GEMV kernels for row-split-k128-v1 packed weights.
//
// out[n] = sum_k W[n,k] * x[k]
// W is stored in the 3-plane row-split layout (low codes, high bits, f16 scales).
// x is bf16 (activation). out is f32.
//
// Decode is bandwidth-bound: each token reads all weights once. The GEMV reads
// the packed codes directly from USM, dequantizes in-register, and accumulates.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>

namespace nsycl::ops {

// f16 -> f32 (handles normal, subnormal, inf, nan).
inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = h & 0x03FFu;
    std::uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign; // +/- 0
        } else {
            float f = static_cast<float>(mant) * 0x1p-15f;
            std::memcpy(&f32, &f, 4);
            f32 |= sign;
        }
    } else if (exp == 0x1F) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f32, 4);
    return result;
}

// bf16 -> f32 (exact for all values).
inline float bf16_to_f32(std::uint16_t b) {
    std::uint32_t f32 = static_cast<std::uint32_t>(b) << 16;
    float result;
    std::memcpy(&result, &f32, 4);
    return result;
}

// Q4 GEMV: 4-bit codes, 2 per byte, 64 values/group, f16 scale/group.
// Dequant: q = (code ^ 8) - 8, range [-8, 7].
inline void gemv_q4(sycl::queue& q,
                    const std::uint8_t* low, const std::uint16_t* scale,
                    const std::uint16_t* x, float* out,
                    int N, int K, int groups_per_row) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int n = id[0];
        const std::uint8_t* row_low   = low + static_cast<std::size_t>(n) * groups_per_row * 32;
        const std::uint16_t* row_scale = scale + static_cast<std::size_t>(n) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            float s = f16_to_f32(row_scale[g]);
            const std::uint8_t* codes = row_low + g * 32;
            float gs = 0.0f;
            for (int p = 0; p < 64; ++p) {
                int k = g * 64 + p;
                if (k >= K) break;
                std::uint8_t byte = codes[p >> 1];
                int code = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                int qv = (code ^ 8) - 8;
                gs += static_cast<float>(qv) * bf16_to_f32(x[k]);
            }
            acc += s * gs;
        }
        out[n] = acc;
    });
}

// Q5 GEMV: 5-bit codes (4 low + 1 high), 64 values/group, f16 scale/group.
// Dequant: u = lo4 | (hi1 << 4); q = (u ^ 16) - 16, range [-16, 15].
inline void gemv_q5(sycl::queue& q,
                    const std::uint8_t* low, const std::uint8_t* high,
                    const std::uint16_t* scale,
                    const std::uint16_t* x, float* out,
                    int N, int K, int groups_per_row) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int n = id[0];
        const std::uint8_t* row_low   = low + static_cast<std::size_t>(n) * groups_per_row * 32;
        const std::uint8_t* row_high  = high + static_cast<std::size_t>(n) * groups_per_row * 8;
        const std::uint16_t* row_scale = scale + static_cast<std::size_t>(n) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            float s = f16_to_f32(row_scale[g]);
            const std::uint8_t* codes = row_low + g * 32;
            const std::uint8_t* hbits = row_high + g * 8;
            float gs = 0.0f;
            for (int p = 0; p < 64; ++p) {
                int k = g * 64 + p;
                if (k >= K) break;
                std::uint8_t byte = codes[p >> 1];
                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                int hi1 = (hbits[p >> 3] >> (p & 7)) & 1;
                int u   = lo4 | (hi1 << 4);
                int qv  = (u ^ 16) - 16;
                gs += static_cast<float>(qv) * bf16_to_f32(x[k]);
            }
            acc += s * gs;
        }
        out[n] = acc;
    });
}

// Q6 GEMV: 6-bit codes (4 low + 2 high), 64 values/group, f16 scale/group.
// Dequant: u = lo4 | (hi2 << 4); q = (u ^ 32) - 32, range [-32, 31].
inline void gemv_q6(sycl::queue& q,
                    const std::uint8_t* low, const std::uint8_t* high,
                    const std::uint16_t* scale,
                    const std::uint16_t* x, float* out,
                    int N, int K, int groups_per_row) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int n = id[0];
        const std::uint8_t* row_low   = low + static_cast<std::size_t>(n) * groups_per_row * 32;
        const std::uint8_t* row_high  = high + static_cast<std::size_t>(n) * groups_per_row * 16;
        const std::uint16_t* row_scale = scale + static_cast<std::size_t>(n) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            float s = f16_to_f32(row_scale[g]);
            const std::uint8_t* codes = row_low + g * 32;
            const std::uint8_t* hbits = row_high + g * 16;
            float gs = 0.0f;
            for (int p = 0; p < 64; ++p) {
                int k = g * 64 + p;
                if (k >= K) break;
                std::uint8_t byte = codes[p >> 1];
                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                int hi2 = (hbits[p >> 1] >> ((p & 1) * 4)) & 0x03;
                int u   = lo4 | (hi2 << 4);
                int qv  = (u ^ 32) - 32;
                gs += static_cast<float>(qv) * bf16_to_f32(x[k]);
            }
            acc += s * gs;
        }
        out[n] = acc;
    });
}

// W8 GEMV: 8-bit codes, 32 values/group, f16 scale/group.
// Dequant: q = (int8)byte, range [-128, 127].
inline void gemv_w8(sycl::queue& q,
                    const std::uint8_t* low, const std::uint16_t* scale,
                    const std::uint16_t* x, float* out,
                    int N, int K, int groups_per_row) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int n = id[0];
        const std::uint8_t* row_low   = low + static_cast<std::size_t>(n) * groups_per_row * 32;
        const std::uint16_t* row_scale = scale + static_cast<std::size_t>(n) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            float s = f16_to_f32(row_scale[g]);
            const std::uint8_t* codes = row_low + g * 32;
            float gs = 0.0f;
            for (int p = 0; p < 32; ++p) {
                int k = g * 32 + p;
                if (k >= K) break;
                int qv = static_cast<std::int8_t>(codes[p]);
                gs += static_cast<float>(qv) * bf16_to_f32(x[k]);
            }
            acc += s * gs;
        }
        out[n] = acc;
    });
}

// Dense BF16 GEMV: out[n] = sum_k W[n,k] * x[k]. W is bf16 [N,K] row-major.
inline void gemv_bf16(sycl::queue& q,
                      const std::uint16_t* w, const std::uint16_t* x, float* out,
                      int N, int K) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int n = id[0];
        const std::uint16_t* row = w + static_cast<std::size_t>(n) * K;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += bf16_to_f32(row[k]) * bf16_to_f32(x[k]);
        }
        out[n] = acc;
    });
}

// W8 row dequant: dequantize one row of a W8G32 tensor to bf16.
// Used for the token embedding lookup (one row per token).
inline void w8_row_dequant(sycl::queue& q,
                           const std::uint8_t* low, const std::uint16_t* scale,
                           std::uint16_t* out, int row, int K, int groups_per_row) {
    q.parallel_for(sycl::range<1>(K), [=](sycl::id<1> id) {
        int k = id[0];
        int g = k / 32;
        int p = k % 32;
        const std::uint8_t* codes = low + static_cast<std::size_t>(row) * groups_per_row * 32 + g * 32;
        const std::uint16_t* row_scale = scale + static_cast<std::size_t>(row) * groups_per_row;
        float s = f16_to_f32(row_scale[g]);
        int qv = static_cast<std::int8_t>(codes[p]);
        float val = static_cast<float>(qv) * s;
        std::uint32_t bits;
        std::memcpy(&bits, &val, 4);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        out[k] = static_cast<std::uint16_t>(bits >> 16);
    });
}

} // namespace nsycl::ops