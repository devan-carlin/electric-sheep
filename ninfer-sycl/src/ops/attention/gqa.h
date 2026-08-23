#pragma once

// nsycl::ops - GQA attention decode (single token).
//
// Qwen3.8-27B: 24 q-heads, 4 kv-heads, head_dim 256, group 6 (kv = q/6).
// KV cache: k, v are bf16 [capacity, kv_heads, head_dim] (token-major).
// q, k, v (current token) are bf16 [heads, head_dim].
// out is f32 [q_heads, head_dim].

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace nsycl::ops::attn {

inline constexpr int kQHeads   = 24;
inline constexpr int kKVHeads  = 4;
inline constexpr int kHeadDim  = 256;
inline constexpr int kGroup    = kQHeads / kKVHeads; // 6
inline constexpr float kAttnScale = 0.0625F;

inline float bf16f(std::uint16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// Single-token GQA decode. One work-item per q-head.
// q: bf16 [24, 256] (current token, rope applied).
// k_cur, v_cur: bf16 [4, 256] (current token k/v, k rope applied).
// cache_k, cache_v: bf16 [capacity, 4, 256] (all prior tokens, token-major).
// seq_len: number of valid tokens INCLUDING the current token (at index seq_len-1).
// out: f32 [24, 256].
inline void gqa_decode(sycl::queue& queue, const std::uint16_t* q, const std::uint16_t* k_cur,
                       const std::uint16_t* v_cur, const std::uint16_t* cache_k,
                       const std::uint16_t* cache_v, int seq_len, float* out) {
    queue.parallel_for(sycl::range<1>(kQHeads), [=](sycl::id<1> id) {
        int hq = id[0];
        int kv = hq / kGroup;

        float qd[kHeadDim];
        for (int d = 0; d < kHeadDim; ++d) {
            qd[d] = bf16f(q[static_cast<std::size_t>(hq) * kHeadDim + d]);
        }

        float m = -1e30f;
        float l = 0.0f;
        float acc[kHeadDim];
        for (int d = 0; d < kHeadDim; ++d) { acc[d] = 0.0f; }

        for (int t = 0; t < seq_len; ++t) {
            const std::uint16_t* krow;
            const std::uint16_t* vrow;
            if (t == seq_len - 1) {
                krow = k_cur + static_cast<std::size_t>(kv) * kHeadDim;
                vrow = v_cur + static_cast<std::size_t>(kv) * kHeadDim;
            } else {
                krow = cache_k + (static_cast<std::size_t>(t) * kKVHeads + kv) * kHeadDim;
                vrow = cache_v + (static_cast<std::size_t>(t) * kKVHeads + kv) * kHeadDim;
            }
            float s = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) { s += qd[d] * bf16f(krow[d]); }
            s *= kAttnScale;
            float m_new = (s > m) ? s : m;
            float scale  = std::expf(m - m_new);
            l = l * scale + std::expf(s - m_new);
            for (int d = 0; d < kHeadDim; ++d) {
                acc[d] = acc[d] * scale + std::expf(s - m_new) * bf16f(vrow[d]);
            }
            m = m_new;
        }
        for (int d = 0; d < kHeadDim; ++d) {
            out[static_cast<std::size_t>(hq) * kHeadDim + d] = acc[d] / l;
        }
    });
}

} // namespace nsycl::ops::attn