#pragma once

// nsycl::ops - Gated DeltaNet (GDN) decode kernels.
//
// Single-token decode. All state is f32 USM.
//
// GDN layer shapes (Qwen3.8-27B):
//   key_heads 16 x 128 (key_dim 2048), value_heads 48 x 128 (value_dim 6144)
//   conv_dim = 2*2048 + 6144 = 10240, conv kernel 4
//   state: [48 v-heads][128 dv][128 dk] f32
//   head map: qk_head = h_v / 3

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace nsycl::ops::gdn {

inline constexpr int kKeyHeads   = 16;
inline constexpr int kValueHeads = 48;
inline constexpr int kHeadDim    = 128;
inline constexpr int kKeyDim     = kKeyHeads * kHeadDim;   // 2048
inline constexpr int kValueDim   = kValueHeads * kHeadDim; // 6144
inline constexpr int kConvDim    = 2 * kKeyDim + kValueDim; // 10240
inline constexpr int kConvKernel = 4;
inline constexpr float kGdnScale = 0.08838834764831845F;
inline constexpr float kL2NormEps = 1.0e-6F;

// Gating: g = -exp(A_log[h]) * softplus(a[h] + dt_bias[h]); beta = sigmoid(b[h]).
// a, b are bf16 [48]; A_log, dt_bias are f32 [48]; g, beta are f32 [48].
inline void gating(sycl::queue& q, const std::uint16_t* a, const std::uint16_t* b,
                   const float* a_log, const float* dt_bias, float* g, float* beta) {
    q.parallel_for(sycl::range<1>(kValueHeads), [=](sycl::id<1> id) {
        int h = id[0];
        auto bf = [](std::uint16_t v) -> float {
            std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
            float r;
            std::memcpy(&r, &bits, 4);
            return r;
        };
        float av = bf(a[h]);
        float bv = bf(b[h]);
        float sp = av + dt_bias[h];
        sp = (sp > 20.0f) ? sp : std::log1pf(std::expf(sp)); // softplus
        g[h]    = -std::expf(a_log[h]) * sp;
        beta[h] = 1.0f / (1.0f + std::expf(-bv));
    });
}

// Causal conv1d (k=4) + SiLU, single token. x is bf16 [C] (current token),
// weight is bf16 [4, C], conv_state is bf16 [3, C] (row 0 = oldest).
// out is bf16 [C]. conv_state is updated in place (shift).
inline void causal_conv1d(sycl::queue& q, const std::uint16_t* x, const std::uint16_t* weight,
                          std::uint16_t* conv_state, std::uint16_t* out, int C) {
    q.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
        int c = id[0];
        auto bf = [](std::uint16_t v) -> float {
            std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
            float r;
            std::memcpy(&r, &bits, 4);
            return r;
        };
        auto to_bf = [](float v) -> std::uint16_t {
            std::uint32_t bits;
            std::memcpy(&bits, &v, 4);
            bits += 0x7FFFu + ((bits >> 16) & 1u);
            return static_cast<std::uint16_t>(bits >> 16);
        };
        float s0 = bf(conv_state[c]);
        float s1 = bf(conv_state[C + c]);
        float s2 = bf(conv_state[2 * C + c]);
        float w0 = bf(weight[c]);
        float w1 = bf(weight[C + c]);
        float w2 = bf(weight[2 * C + c]);
        float w3 = bf(weight[3 * C + c]);
        float x0 = bf(x[c]);
        float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x0;
        float silu = acc / (1.0f + std::expf(-acc));
        out[c] = to_bf(silu);
        // shift state: s0 <- s1, s1 <- s2, s2 <- x0
        conv_state[c]      = to_bf(s1);
        conv_state[C + c]  = to_bf(s2);
        conv_state[2 * C + c] = to_bf(x0);
    });
}

// L2-normalize q and k per head (16 heads x 128). In place, bf16.
inline void normalize_qk(sycl::queue& queue, std::uint16_t* q, std::uint16_t* k) {
    queue.parallel_for(sycl::range<1>(2 * kKeyHeads), [=](sycl::id<1> id) {
        int h = id[0];
        std::uint16_t* ptr = (h < kKeyHeads) ? (q + static_cast<std::size_t>(h) * kHeadDim)
                                             : (k + static_cast<std::size_t>(h - kKeyHeads) * kHeadDim);
        float ss = 0.0f;
        float vals[kHeadDim];
        for (int i = 0; i < kHeadDim; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(ptr[i]) << 16;
            float v;
            std::memcpy(&v, &bits, 4);
            vals[i] = v;
            ss += v * v;
        }
        float inv = 1.0f / std::sqrtf(ss + kL2NormEps);
        for (int i = 0; i < kHeadDim; ++i) {
            float v = vals[i] * inv;
            std::uint32_t bits;
            std::memcpy(&bits, &v, 4);
            bits += 0x7FFFu + ((bits >> 16) & 1u);
            ptr[i] = static_cast<std::uint16_t>(bits >> 16);
        }
    });
}

// GDN recurrent decode (single token).
// q, k: bf16 [16, 128] (normalized). v: bf16 [48, 128]. g, beta: f32 [48].
// state: f32 [48, 128, 128] (updated in place). out: f32 [48, 128].
inline void recurrent(sycl::queue& queue, const std::uint16_t* q, const std::uint16_t* k,
                      const std::uint16_t* v, const float* g, const float* beta,
                      float* state, float* out) {
    queue.parallel_for(sycl::range<1>(kValueHeads * kHeadDim), [=](sycl::id<1> id) {
        int hv = id[0] / kHeadDim;
        int r  = id[0] % kHeadDim;
        int qk = hv / 3;
        float alpha = std::expf(g[hv]);
        float beta_v = beta[hv];

        // load k[qk] (128)
        float kv[kHeadDim];
        for (int c = 0; c < kHeadDim; ++c) {
            std::uint32_t bits = static_cast<std::uint32_t>(k[static_cast<std::size_t>(qk) * kHeadDim + c]) << 16;
            std::memcpy(&kv[c], &bits, 4);
        }
        float* st = state + (static_cast<std::size_t>(hv) * kHeadDim + r) * kHeadDim;
        float partial = 0.0f;
        for (int c = 0; c < kHeadDim; ++c) { partial += st[c] * kv[c]; }
        float v_r;
        {
            std::uint32_t bits = static_cast<std::uint32_t>(v[static_cast<std::size_t>(hv) * kHeadDim + r]) << 16;
            std::memcpy(&v_r, &bits, 4);
        }
        float delta = beta_v * (v_r - alpha * partial);
        for (int c = 0; c < kHeadDim; ++c) {
            st[c] = alpha * st[c] + delta * kv[c];
        }
        // readout
        float qv[kHeadDim];
        for (int c = 0; c < kHeadDim; ++c) {
            std::uint32_t bits = static_cast<std::uint32_t>(q[static_cast<std::size_t>(qk) * kHeadDim + c]) << 16;
            std::memcpy(&qv[c], &bits, 4);
        }
        float attn = 0.0f;
        for (int c = 0; c < kHeadDim; ++c) { attn += st[c] * qv[c]; }
        out[static_cast<std::size_t>(hv) * kHeadDim + r] = attn * kGdnScale;
    });
}

} // namespace nsycl::ops::gdn