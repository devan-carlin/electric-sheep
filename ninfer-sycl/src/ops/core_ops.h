#pragma once

// nsycl::ops - core elementwise, norm, and rope kernels for the decode path.
//
// All operate on bf16 (or f32) USM pointers. Single-token decode shapes.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace nsycl::ops {

inline float f16_to_f32(std::uint16_t h);
inline float bf16_to_f32(std::uint16_t b);

// f32 -> bf16 (round-to-nearest-even).
inline std::uint16_t f32_to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight. x, weight, out are bf16.
// n is the row length.
inline void rmsnorm(sycl::queue& q, const std::uint16_t* x, const std::uint16_t* weight,
                    float eps, std::uint16_t* out, int n) {
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        float ss = 0.0f;
        for (int i = 0; i < n; ++i) {
            float v = bf16_to_f32(x[i]);
            ss += v * v;
        }
        float inv = 1.0f / std::sqrtf(ss / static_cast<float>(n) + eps);
        for (int i = 0; i < n; ++i) {
            // unit_offset: stored weight is (real - 1); effective = weight + 1.0.
            float w = bf16_to_f32(weight[i]) + 1.0f;
            float v = bf16_to_f32(x[i]) * inv * w;
            out[i] = f32_to_bf16(v);
        }
    });
}

// Per-head RMSNorm: apply rmsnorm (weight [head_dim]) to each of `heads` rows
// of length head_dim. x, out are bf16 [heads*head_dim].
inline void rmsnorm_heads(sycl::queue& q, const std::uint16_t* x, const std::uint16_t* weight,
                          float eps, std::uint16_t* out, int heads, int head_dim) {
    q.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> id) {
        int h = id[0];
        const std::uint16_t* xh = x + static_cast<std::size_t>(h) * head_dim;
        std::uint16_t* out_h    = out + static_cast<std::size_t>(h) * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float v = bf16_to_f32(xh[i]);
            ss += v * v;
        }
        float inv = 1.0f / std::sqrtf(ss / static_cast<float>(head_dim) + eps);
        for (int i = 0; i < head_dim; ++i) {
            // unit_offset: stored weight is (real - 1); effective = weight + 1.0.
            float w = bf16_to_f32(weight[i]) + 1.0f;
            float v = bf16_to_f32(xh[i]) * inv * w;
            out_h[i] = f32_to_bf16(v);
        }
    });
}

// Gated RMSNorm (per head): out = (o / rms(o)) * weight * SiLU(z).
// o, z are f32 [heads*dim], weight is bf16 [dim], out is f32 [heads*dim].
// Matches the reference: no unit offset on weight; gate is SiLU(z) = z*sigmoid(z).
inline void gated_rmsnorm(sycl::queue& q, const float* o, const std::uint16_t* weight,
                          const float* z, float eps, float* out, int heads, int dim) {
    q.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> id) {
        int h = id[0];
        const float* oh = o + static_cast<std::size_t>(h) * dim;
        const float* zh = z + static_cast<std::size_t>(h) * dim;
        float* out_h    = out + static_cast<std::size_t>(h) * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) { ss += oh[i] * oh[i]; }
        float inv = 1.0f / std::sqrtf(ss / static_cast<float>(dim) + eps);
        for (int i = 0; i < dim; ++i) {
            float w = bf16_to_f32(weight[i]);
            float silu_z = zh[i] / (1.0f + std::expf(-zh[i]));
            out_h[i] = oh[i] * inv * w * silu_z;
        }
    });
}

// RoPE (rotate-half) on the first `rot` dims of each head.
// q: bf16 [q_heads*head_dim], k: bf16 [k_heads*head_dim]. Single token.
inline void rope(sycl::queue& queue, std::uint16_t* q, int q_heads, std::uint16_t* k, int k_heads,
                 int head_dim, int rot, float theta, int position) {
    auto apply = [&](std::uint16_t* ptr, int heads) {
        queue.parallel_for(sycl::range<1>(heads * (rot / 2)), [=](sycl::id<1> id) {
            int idx = id[0];
            int h   = idx / (rot / 2);
            int i   = idx % (rot / 2);
            float freq = std::powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rot));
            float ang  = static_cast<float>(position) * freq;
            float s    = std::sinf(ang);
            float c    = std::cosf(ang);
            std::uint16_t* ph = ptr + static_cast<std::size_t>(h) * head_dim;
            float x0 = bf16_to_f32(ph[i]);
            float x1 = bf16_to_f32(ph[i + rot / 2]);
            ph[i]         = f32_to_bf16(x0 * c - x1 * s);
            ph[i + rot/2] = f32_to_bf16(x1 * c + x0 * s);
        });
    };
    apply(q, q_heads);
    apply(k, k_heads);
}

// SiLU: out = x * sigmoid(x). f32 in/out.
inline void silu(sycl::queue& q, const float* x, float* out, int n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        float v = x[id[0]];
        out[id[0]] = v / (1.0f + std::expf(-v));
    });
}

// SiLU-mul: out = silu(a) * b. f32 in/out.
inline void silu_mul(sycl::queue& q, const float* a, const float* b, float* out, int n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        float av = a[id[0]];
        out[id[0]] = (av / (1.0f + std::expf(-av))) * b[id[0]];
    });
}

// Sigmoid-mul: out = sigmoid(a) * b. f32 in/out.
inline void sigmoid_mul(sycl::queue& q, const float* a, const float* b, float* out, int n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        float av = a[id[0]];
        out[id[0]] = (1.0f / (1.0f + std::expf(-av))) * b[id[0]];
    });
}

// f32 -> bf16 in place (convert a f32 buffer to bf16).
inline void f32_to_bf16_buf(sycl::queue& q, const float* x, std::uint16_t* out, int n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        out[id[0]] = f32_to_bf16(x[id[0]]);
    });
}

} // namespace nsycl::ops