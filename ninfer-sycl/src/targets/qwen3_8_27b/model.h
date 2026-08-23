#pragma once

// nsycl::model - Qwen3.8-27B text model: 64-layer decode forward pass.
//
// Single-token greedy decode. MTP0 baseline (no speculative drafting).
// Layer map: full-attention at layers 3,7,...,63 (16); GDN elsewhere (48).

#include "targets/qwen3_8_27b/weights.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nsycl::model {

struct ModelConfig {
    static constexpr int hidden       = 5120;
    static constexpr int layers       = 64;
    static constexpr int intermediate = 17408;
    static constexpr int vocab        = 248320;
    // Only token IDs in [0, token_domain) are valid; rows [token_domain, vocab)
    // are padding in the output matrix and must be excluded from argmax.
    static constexpr int token_domain = 248077;
    static constexpr int query_heads  = 24;
    static constexpr int kv_heads     = 4;
    static constexpr int head_dim     = 256;
    static constexpr int rotary_dim   = 64;
    static constexpr int query_size   = query_heads * head_dim; // 6144
    static constexpr int kv_size      = kv_heads * head_dim;    // 1024
    static constexpr int gdn_key_heads  = 16;
    static constexpr int gdn_value_heads = 48;
    static constexpr int gdn_head_dim   = 128;
    static constexpr int gdn_key_dim    = gdn_key_heads * gdn_head_dim;   // 2048
    static constexpr int gdn_value_dim  = gdn_value_heads * gdn_head_dim; // 6144
    static constexpr int gdn_conv_dim   = 2 * gdn_key_dim + gdn_value_dim; // 10240
    static constexpr int full_attn_layers = 16;
    static constexpr int gdn_layers     = 48;
    static constexpr int max_seq_len    = 262144;
    static constexpr float rms_eps      = 1.0e-6F;
    static constexpr float rope_theta   = 1.0e7F;

    static constexpr bool is_full_attention(int layer) { return (layer + 1) % 4 == 0; }
    static constexpr int full_attn_index(int layer) { return (layer + 1) / 4 - 1; }
    static constexpr int gdn_index(int layer) { return layer - (layer + 1) / 4; }
};

class Model {
public:
    explicit Model(sycl::queue& q, const std::filesystem::path& artifact_path);
    ~Model();

    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    // Embed a prompt (token ids) and run the forward pass for each, filling the
    // KV cache and GDN state. Computes the first generated token from the last
    // prompt token's hidden state and returns it.
    int prefill(const std::vector<int>& tokens, sycl::queue& q);

    // Decode one token: embed the last generated token, run the full forward
    // pass, return the argmax next token.
    int decode(sycl::queue& q);

    int seq_len() const { return seq_len_; }

private:
    void attn_layer(int fidx, std::uint16_t* x, sycl::queue& q);
    void gdn_layer(int gidx, std::uint16_t* x, sycl::queue& q);
    void mlp_layer(int layer, std::uint16_t* x, sycl::queue& q);
    int compute_logits(sycl::queue& q);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    int seq_len_ = 0;
    int last_token_ = 0;
};

} // namespace nsycl::model