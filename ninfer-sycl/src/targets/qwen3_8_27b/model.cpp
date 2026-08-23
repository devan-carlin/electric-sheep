#include "targets/qwen3_8_27b/model.h"

#include "core/verbose.h"
#include "ops/attention/gqa.h"
#include "ops/core_ops.h"
#include "ops/gdn/gdn.h"
#include "ops/linear/gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nsycl::model {
namespace {

using C = ModelConfig;

std::uint16_t to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// f32 -> bf16 (device).
void f32_to_bf16_(sycl::queue& q, const float* x, std::uint16_t* out, int n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        out[id[0]] = to_bf16(x[id[0]]);
    });
}

} // namespace

struct Model::Impl {
    Weights weights;
    sycl::queue* q = nullptr;

    // Workspace (USM).
    std::uint16_t* hidden = nullptr;   // [hidden] bf16
    std::uint16_t* h_norm = nullptr;   // [hidden] bf16
    float* qkv_f = nullptr;            // [7168] f32 (attn q/k gemv out)
    float* gv_f  = nullptr;            // [7168] f32 (attn gate/v gemv out)
    std::uint16_t* q_ptr = nullptr;    // [6144] bf16
    std::uint16_t* k_ptr = nullptr;    // [1024] bf16
    std::uint16_t* gate_ptr = nullptr; // [6144] bf16
    std::uint16_t* v_ptr = nullptr;    // [1024] bf16
    std::uint16_t* qn     = nullptr;   // [6144] bf16 (attn q norm)
    std::uint16_t* kn     = nullptr;   // [1024] bf16 (attn k norm)
    float* attn_out       = nullptr;   // [6144] f32
    std::uint16_t* attn_out_bf = nullptr; // [6144] bf16
    float* mlp_gate_up    = nullptr;   // [34816] f32
    float* mlp_act        = nullptr;   // [17408] f32
    std::uint16_t* mlp_act_bf = nullptr; // [17408] bf16
    float* mlp_down       = nullptr;   // [hidden] f32
    float* logits         = nullptr;   // [vocab] f32
    std::uint16_t* emb    = nullptr;   // [hidden] bf16

    // GDN workspace.
    float* gdn_qkv_f = nullptr;        // [4096] f32 (q/k gemv out)
    float* gdn_vz_f  = nullptr;        // [12288] f32 (v/z gemv out)
    std::uint16_t* gdn_qkv = nullptr;  // [10240] bf16 (conv input/output)
    std::uint16_t* gdn_q   = nullptr;  // [2048] bf16
    std::uint16_t* gdn_k   = nullptr;  // [2048] bf16
    std::uint16_t* gdn_v   = nullptr;  // [6144] bf16
    std::uint16_t* gdn_z   = nullptr;  // [6144] bf16
    float* gdn_g          = nullptr;   // [48] f32
    float* gdn_beta       = nullptr;   // [48] f32
    float* gdn_a          = nullptr;   // [48] f32
    float* gdn_b          = nullptr;   // [48] f32
    std::uint16_t* gdn_a_bf = nullptr; // [48] bf16
    std::uint16_t* gdn_b_bf = nullptr; // [48] bf16
    float* gdn_out        = nullptr;   // [6144] f32
    std::uint16_t* gdn_out_bf = nullptr; // [6144] bf16
    float* gdn_on         = nullptr;   // [6144] f32

    // KV cache (token-major): [max_seq, kv_heads, head_dim] bf16.
    std::uint16_t* cache_k = nullptr;
    std::uint16_t* cache_v = nullptr;

    // GDN state (persistent across tokens).
    float* gdn_state = nullptr;   // [48 layers][48 heads][128][128] f32
    std::uint16_t* gdn_conv = nullptr; // [48 layers][3][10240] bf16

    std::uint8_t* usm_base = nullptr;
    std::size_t usm_bytes = 0;
};

Model::Model(sycl::queue& q, const std::filesystem::path& artifact_path)
    : impl_(std::make_unique<Impl>()) {
    impl_->q = &q;
    impl_->weights = Weights(q, artifact_path);

    std::size_t bytes = 0;
    auto add = [&](std::size_t n, std::size_t sz) {
        bytes = (bytes + 255) & ~std::size_t(255);
        bytes += n * sz;
    };
    add(C::hidden, 2);   // hidden
    add(C::hidden, 2);   // h_norm
    add(7168, 4);        // qkv_f
    add(7168, 4);        // gv_f
    add(C::query_size, 2); // q_ptr
    add(C::kv_size, 2);    // k_ptr
    add(C::query_size, 2); // gate_ptr
    add(C::kv_size, 2);    // v_ptr
    add(C::query_size, 2); // qn
    add(C::kv_size, 2);    // kn
    add(C::query_size, 4); // attn_out
    add(C::query_size, 2); // attn_out_bf
    add(34816, 4);         // mlp_gate_up
    add(C::intermediate, 4); // mlp_act
    add(C::intermediate, 2); // mlp_act_bf
    add(C::hidden, 4);     // mlp_down
    add(C::vocab, 4);      // logits
    add(C::hidden, 2);     // emb
    add(4096, 4);          // gdn_qkv_f
    add(12288, 4);         // gdn_vz_f
    add(C::gdn_conv_dim, 2); // gdn_qkv
    add(C::gdn_key_dim, 2);  // gdn_q
    add(C::gdn_key_dim, 2);  // gdn_k
    add(C::gdn_value_dim, 2); // gdn_v
    add(C::gdn_value_dim, 2); // gdn_z
    add(48, 4);           // gdn_g
    add(48, 4);           // gdn_beta
    add(48, 4);           // gdn_a
    add(48, 4);           // gdn_b
    add(48, 2);           // gdn_a_bf
    add(48, 2);           // gdn_b_bf
    add(C::gdn_value_dim, 4); // gdn_out
    add(C::gdn_value_dim, 2); // gdn_out_bf
    add(C::gdn_value_dim, 4); // gdn_on
    add(std::size_t(C::full_attn_layers) * C::max_seq_len * C::kv_heads * C::head_dim, 2); // cache_k (per-layer)
    add(std::size_t(C::full_attn_layers) * C::max_seq_len * C::kv_heads * C::head_dim, 2); // cache_v (per-layer)
    add(std::size_t(C::gdn_layers) * 48 * 128 * 128, 4);  // gdn_state
    add(std::size_t(C::gdn_layers) * 3 * C::gdn_conv_dim, 2); // gdn_conv

    impl_->usm_bytes = bytes;
    // Shared (system) USM: coherent across kernels. Device USM on this Level
    // Zero driver has a kernel-to-kernel write-visibility gap (first reader of
    // a freshly-written buffer sees stale data), so the workspace uses shared
    // memory. Weights stay in device USM (read-only, bandwidth-critical).
    impl_->usm_base = sycl::malloc_shared<std::uint8_t>(bytes, q);

    std::size_t off = 0;
    auto carve = [&](auto* ptr, std::size_t n, std::size_t sz) {
        off = (off + 255) & ~std::size_t(255);
        *ptr = reinterpret_cast<std::remove_reference_t<decltype(*ptr)>>(impl_->usm_base + off);
        off += n * sz;
    };
    carve(&impl_->hidden, C::hidden, 2);
    carve(&impl_->h_norm, C::hidden, 2);
    carve(&impl_->qkv_f, 7168, 4);
    carve(&impl_->gv_f, 7168, 4);
    carve(&impl_->q_ptr, C::query_size, 2);
    carve(&impl_->k_ptr, C::kv_size, 2);
    carve(&impl_->gate_ptr, C::query_size, 2);
    carve(&impl_->v_ptr, C::kv_size, 2);
    carve(&impl_->qn, C::query_size, 2);
    carve(&impl_->kn, C::kv_size, 2);
    carve(&impl_->attn_out, C::query_size, 4);
    carve(&impl_->attn_out_bf, C::query_size, 2);
    carve(&impl_->mlp_gate_up, 34816, 4);
    carve(&impl_->mlp_act, C::intermediate, 4);
    carve(&impl_->mlp_act_bf, C::intermediate, 2);
    carve(&impl_->mlp_down, C::hidden, 4);
    carve(&impl_->logits, C::vocab, 4);
    carve(&impl_->emb, C::hidden, 2);
    carve(&impl_->gdn_qkv_f, 4096, 4);
    carve(&impl_->gdn_vz_f, 12288, 4);
    carve(&impl_->gdn_qkv, C::gdn_conv_dim, 2);
    carve(&impl_->gdn_q, C::gdn_key_dim, 2);
    carve(&impl_->gdn_k, C::gdn_key_dim, 2);
    carve(&impl_->gdn_v, C::gdn_value_dim, 2);
    carve(&impl_->gdn_z, C::gdn_value_dim, 2);
    carve(&impl_->gdn_g, 48, 4);
    carve(&impl_->gdn_beta, 48, 4);
    carve(&impl_->gdn_a, 48, 4);
    carve(&impl_->gdn_b, 48, 4);
    carve(&impl_->gdn_a_bf, 48, 2);
    carve(&impl_->gdn_b_bf, 48, 2);
    carve(&impl_->gdn_out, C::gdn_value_dim, 4);
    carve(&impl_->gdn_out_bf, C::gdn_value_dim, 2);
    carve(&impl_->gdn_on, C::gdn_value_dim, 4);
    carve(&impl_->cache_k, std::size_t(C::full_attn_layers) * C::max_seq_len * C::kv_heads * C::head_dim, 2);
    carve(&impl_->cache_v, std::size_t(C::full_attn_layers) * C::max_seq_len * C::kv_heads * C::head_dim, 2);
    carve(&impl_->gdn_state, std::size_t(C::gdn_layers) * 48 * 128 * 128, 4);
    carve(&impl_->gdn_conv, std::size_t(C::gdn_layers) * 3 * C::gdn_conv_dim, 2);

    // Materialize + zero the entire workspace USM buffer. A full memset forces
    // the driver to back every page, which avoids a first-touch issue where the
    // first kernel that writes to an unmaterialized USM region reads back zeros.
    q.memset(impl_->usm_base, 0, impl_->usm_bytes);
    q.wait();
    {
        std::vector<std::uint8_t> check(1024);
        q.memcpy(check.data(), impl_->usm_base, 1024);
        q.wait();
        bool all_zero = true;
        for (auto b : check) if (b != 0) { all_zero = false; break; }
        NINFER_VERBOSE("Model: workspace readback after memset: %s", all_zero ? "all zero" : "NON-ZERO");
    }
    NINFER_VERBOSE("Model: workspace %.3f GiB", (double)impl_->usm_bytes / (1024.0 * 1024.0 * 1024.0));
}

Model::~Model() = default;

void Model::attn_layer(int fidx, std::uint16_t* x, sycl::queue& q) {
    auto* I = impl_.get();
    std::string base = "text/layers/" + std::to_string(4 * (fidx + 1) - 1);

    // 1. h = rmsnorm(x, input_norm)
    ops::rmsnorm(q, x, I->weights.bf16(base + "/input_norm"), C::rms_eps, I->h_norm, C::hidden);

    // 2. q,k = query_key_proj(h); gate,v = gate_value_proj(h)
    I->weights.gemv(base + "/attention/query_key", I->h_norm, I->qkv_f, q);
    I->weights.gemv(base + "/attention/gate_value", I->h_norm, I->gv_f, q);
    // qkv_f: [0:6144]=q, [6144:7168]=k. gv_f: [0:6144]=gate, [6144:7168]=v.
    f32_to_bf16_(q, I->qkv_f, I->q_ptr, C::query_size);
    f32_to_bf16_(q, I->qkv_f + C::query_size, I->k_ptr, C::kv_size);
    f32_to_bf16_(q, I->gv_f, I->gate_ptr, C::query_size);
    f32_to_bf16_(q, I->gv_f + C::query_size, I->v_ptr, C::kv_size);

    // 3. qn = rmsnorm_heads(q, query_norm); kn = rmsnorm_heads(k, key_norm)
    ops::rmsnorm_heads(q, I->q_ptr, I->weights.bf16(base + "/attention/query_norm"),
                       C::rms_eps, I->qn, C::query_heads, C::head_dim);
    ops::rmsnorm_heads(q, I->k_ptr, I->weights.bf16(base + "/attention/key_norm"),
                       C::rms_eps, I->kn, C::kv_heads, C::head_dim);

    // 4. rope(qn, kn)
    ops::rope(q, I->qn, C::query_heads, I->kn, C::kv_heads, C::head_dim, C::rotary_dim,
              C::rope_theta, seq_len_);

    // 5. Write current k/v into this layer's KV cache at position seq_len_.
    {
        const std::size_t layer_stride = std::size_t(C::max_seq_len) * C::kv_heads * C::head_dim;
        std::uint16_t* ck_base = I->cache_k + std::size_t(fidx) * layer_stride;
        std::uint16_t* cv_base = I->cache_v + std::size_t(fidx) * layer_stride;
        std::size_t pos = seq_len_;
        std::uint16_t* ck = ck_base + pos * C::kv_heads * C::head_dim;
        std::uint16_t* cv = cv_base + pos * C::kv_heads * C::head_dim;
        q.memcpy(ck, I->kn, C::kv_size * 2);
        q.memcpy(cv, I->v_ptr, C::kv_size * 2);
    }

    // 6. a = gqa_decode(qn, this layer's cache)
    {
        const std::size_t layer_stride = std::size_t(C::max_seq_len) * C::kv_heads * C::head_dim;
        std::uint16_t* ck_base = I->cache_k + std::size_t(fidx) * layer_stride;
        std::uint16_t* cv_base = I->cache_v + std::size_t(fidx) * layer_stride;
        ops::attn::gqa_decode(q, I->qn, I->kn, I->v_ptr, ck_base, cv_base, seq_len_ + 1,
                              I->attn_out);
    }

    // 7. a = sigmoid(gate) * a
    q.parallel_for(sycl::range<1>(C::query_size), [=](sycl::id<1> id) {
        float g = from_bf16(I->gate_ptr[id[0]]);
        I->attn_out[id[0]] = (1.0f / (1.0f + std::expf(-g))) * I->attn_out[id[0]];
    });

    if (nsycl::verbose_enabled() && nsycl::log_layer(4 * (fidx + 1) - 1)) {
        auto rmsbf3 = [&](const std::uint16_t* p, int n) {
            std::vector<std::uint16_t> h(n);
            q.memcpy(h.data(), p, n * 2);
            q.wait();
            double ss = 0.0;
            for (auto v : h) { float f = from_bf16(v); ss += (double)f * f; }
            return std::sqrt(ss / n);
        };
        auto rmsf3 = [&](const float* p, int n) {
            std::vector<float> h(n);
            q.memcpy(h.data(), p, n * 4);
            q.wait();
            double ss = 0.0;
            for (float v : h) ss += (double)v * v;
            return std::sqrt(ss / n);
        };
        std::fprintf(stderr, "[verbose] ATTN layer=%d qn rms=%.4f kn rms=%.4f v rms=%.4f gate rms=%.4f attn_out rms=%.4f\n",
                     4 * (fidx + 1) - 1,
                     rmsbf3(I->qn, C::query_size),
                     rmsbf3(I->kn, C::kv_size),
                     rmsbf3(I->v_ptr, C::kv_size),
                     rmsbf3(I->gate_ptr, C::query_size),
                     rmsf3(I->attn_out, C::query_size));
    }

    // Deep attention dump (level 4): recompute head-0 attention on the host in
    // double precision from the read-back qn + KV cache, and compare against the
    // kernel output. Also dumps the raw scores + softmax weights so the scores
    // can be checked independently (numpy).
    if (nsycl::verbose_level() >= 4 && nsycl::log_layer(4 * (fidx + 1) - 1) &&
        nsycl::log_token(seq_len_)) {
        const int layer = 4 * (fidx + 1) - 1;
        const int S = seq_len_ + 1; // number of valid tokens (incl. current)
        auto rmsbf3_local = [&](const std::uint16_t* p, int n) {
            std::vector<std::uint16_t> h(n);
            q.memcpy(h.data(), p, n * 2);
            q.wait();
            double ss = 0.0;
            for (auto v : h) { float f = from_bf16(v); ss += (double)f * f; }
            return std::sqrt(ss / n);
        };
        // Read back qn head 0, cache_k/v kv-head 0 for all S positions, kernel
        // attn_out head 0, and gate head 0.
        std::vector<std::uint16_t> qn0(C::head_dim);
        std::vector<std::uint16_t> ck0(std::size_t(S) * C::head_dim);
        std::vector<std::uint16_t> cv0(std::size_t(S) * C::head_dim);
        std::vector<float> aout0(C::head_dim);
        std::vector<std::uint16_t> gate0(C::head_dim);
        q.memcpy(qn0.data(), I->qn, C::head_dim * 2);
        // cache is token-major [pos, kv_heads, head_dim]; kv head 0 is the first
        // head_dim of each token row. Per-layer base offset by fidx.
        const std::size_t layer_stride = std::size_t(C::max_seq_len) * C::kv_heads * C::head_dim;
        std::uint16_t* ck_base = I->cache_k + std::size_t(fidx) * layer_stride;
        std::uint16_t* cv_base = I->cache_v + std::size_t(fidx) * layer_stride;
        for (int t = 0; t < S; ++t) {
            q.memcpy(ck0.data() + t * C::head_dim,
                     ck_base + (std::size_t(t) * C::kv_heads) * C::head_dim,
                     C::head_dim * 2);
            q.memcpy(cv0.data() + t * C::head_dim,
                     cv_base + (std::size_t(t) * C::kv_heads) * C::head_dim,
                     C::head_dim * 2);
        }
        q.memcpy(aout0.data(), I->attn_out, C::head_dim * 4);
        q.memcpy(gate0.data(), I->gate_ptr, C::head_dim * 2);
        q.wait();
        // Host recompute (double).
        std::vector<double> qd(C::head_dim), scores(S);
        for (int d = 0; d < C::head_dim; ++d) qd[d] = from_bf16(qn0[d]);
        double mx = -1e30;
        for (int t = 0; t < S; ++t) {
            double s = 0.0;
            for (int d = 0; d < C::head_dim; ++d)
                s += qd[d] * from_bf16(ck0[std::size_t(t) * C::head_dim + d]);
            s *= 0.0625;
            scores[t] = s;
            if (s > mx) mx = s;
        }
        double l = 0.0;
        std::vector<double> w(S);
        for (int t = 0; t < S; ++t) { w[t] = std::exp(scores[t] - mx); l += w[t]; }
        for (int t = 0; t < S; ++t) w[t] /= l;
        std::vector<double> host_aout(C::head_dim, 0.0);
        for (int d = 0; d < C::head_dim; ++d)
            for (int t = 0; t < S; ++t)
                host_aout[d] += w[t] * from_bf16(cv0[std::size_t(t) * C::head_dim + d]);
        // Gate sigmoid (host).
        double gmin = 1e30, gmax = -1e30, gss = 0;
        for (int d = 0; d < C::head_dim; ++d) {
            double g = from_bf16(gate0[d]);
            double sg = 1.0 / (1.0 + std::exp(-g));
            gmin = std::min(gmin, sg); gmax = std::max(gmax, sg); gss += sg * sg;
        }
        // Apply the gate to the host recompute so it matches the kernel output
// (which is gated in step 7).
        std::vector<double> host_gated(C::head_dim);
        for (int d = 0; d < C::head_dim; ++d) {
            double g = from_bf16(gate0[d]);
            host_gated[d] = (1.0 / (1.0 + std::exp(-g))) * host_aout[d];
        }
        // Kernel vs host (both gated) head 0.
        double maxdiff = 0.0;
        for (int d = 0; d < C::head_dim; ++d)
            maxdiff = std::max(maxdiff, std::abs((double)aout0[d] - host_gated[d]));
        std::fprintf(stderr, "[deep] ATTN layer=%d pos=%d S=%d\n", layer, seq_len_, S);
        std::fprintf(stderr, "[deep]   qn0[0:8]=");
        for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.4f ", from_bf16(qn0[d]));
        std::fprintf(stderr, "\n[deep]   scores[0:%d]=", S);
        for (int t = 0; t < S; ++t) std::fprintf(stderr, "%.4f ", scores[t]);
        std::fprintf(stderr, "\n[deep]   softmax[0:%d]=", S);
        for (int t = 0; t < S; ++t) std::fprintf(stderr, "%.5f ", w[t]);
        std::fprintf(stderr, "\n[deep]   gate_sigmoid head0: min=%.4f max=%.4f rms=%.4f\n",
                     gmin, gmax, std::sqrt(gss / C::head_dim));
        std::fprintf(stderr, "[deep]   kernel aout0[0:8]=");
        for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.5f ", aout0[d]);
        std::fprintf(stderr, "\n[deep]   host   aout0[0:8]=");
        for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.5f ", host_gated[d]);
        std::fprintf(stderr, "\n[deep]   kernel-vs-host maxdiff=%.6f\n", maxdiff);
        // Cache sanity: is the current-position cache row non-zero and equal to
        // the just-computed kn/v?
        double ck_cur_rms = 0.0;
        for (int d = 0; d < C::head_dim; ++d) {
            double v = from_bf16(ck0[std::size_t(S - 1) * C::head_dim + d]);
            ck_cur_rms += v * v;
        }
        std::fprintf(stderr, "[deep]   cache_k[pos=%d] rms=%.5f (kn rms=%.5f)\n",
                     seq_len_, std::sqrt(ck_cur_rms / C::head_dim),
                     rmsbf3_local(I->kn, C::kv_size));
    }

    // 8. x = x + output_proj(a)
    f32_to_bf16_(q, I->attn_out, I->attn_out_bf, C::query_size);
    I->weights.gemv(base + "/attention/output", I->attn_out_bf, I->mlp_down, q);
    q.parallel_for(sycl::range<1>(C::hidden), [=](sycl::id<1> id) {
        x[id[0]] = to_bf16(from_bf16(x[id[0]]) + I->mlp_down[id[0]]);
    });
}

void Model::gdn_layer(int gidx, std::uint16_t* x, sycl::queue& q) {
    auto* I = impl_.get();
    // Layout is [G,G,G,A] repeated: GDN gidx maps to layer 4*(gidx/3) + (gidx%3).
    int layer = 4 * (gidx / 3) + (gidx % 3);
    std::string base = "text/layers/" + std::to_string(layer) + "/gdn";

    // 1. h = rmsnorm(x, input_norm)
    std::string inorm = "text/layers/" + std::to_string(layer) + "/input_norm";
    ops::rmsnorm(q, x, I->weights.bf16(inorm), C::rms_eps, I->h_norm, C::hidden);

    // 2. q,k = query_key_proj(h); v,z = value_z_proj(h)
    I->weights.gemv(base + "/query_key", I->h_norm, I->gdn_qkv_f, q);
    I->weights.gemv(base + "/value_z", I->h_norm, I->gdn_vz_f, q);
    // gdn_qkv_f: [0:2048]=q, [2048:4096]=k. gdn_vz_f: [0:6144]=v, [6144:12288]=z.
    f32_to_bf16_(q, I->gdn_qkv_f, I->gdn_q, C::gdn_key_dim);
    f32_to_bf16_(q, I->gdn_qkv_f + C::gdn_key_dim, I->gdn_k, C::gdn_key_dim);
    f32_to_bf16_(q, I->gdn_vz_f, I->gdn_v, C::gdn_value_dim);
    // z stays f32 in gdn_vz_f + gdn_value_dim (gated_rmsnorm takes f32 z).

    if (nsycl::verbose_enabled() && nsycl::log_layer(layer)) {
        // Pre-conv dump: raw GEMV outputs (before causal_conv1d overwrites gdn_q/k/v).
        auto rmsbf2 = [&](const std::uint16_t* p, int n) {
            std::vector<std::uint16_t> h(n);
            q.memcpy(h.data(), p, n * 2);
            q.wait();
            double ss = 0.0;
            for (auto v : h) { float f = from_bf16(v); ss += (double)f * f; }
            return std::sqrt(ss / n);
        };
        auto rmsf2 = [&](const float* p, int n) {
            std::vector<float> h(n);
            q.memcpy(h.data(), p, n * 4);
            q.wait();
            double ss = 0.0;
            for (float v : h) ss += (double)v * v;
            return std::sqrt(ss / n);
        };
        std::fprintf(stderr, "[verbose] GDN layer=%d PRE-CONV h_norm rms=%.4f | f32 qkv: q=%.4f k=%.4f | f32 vz: v=%.4f z=%.4f | bf16 q=%.4f k=%.4f v=%.4f\n",
                     layer,
                     rmsbf2(I->h_norm, C::hidden),
                     rmsf2(I->gdn_qkv_f, C::gdn_key_dim),
                     rmsf2(I->gdn_qkv_f + C::gdn_key_dim, C::gdn_key_dim),
                     rmsf2(I->gdn_vz_f, C::gdn_value_dim),
                     rmsf2(I->gdn_vz_f + C::gdn_value_dim, C::gdn_value_dim),
                     rmsbf2(I->gdn_q, C::gdn_key_dim),
                     rmsbf2(I->gdn_k, C::gdn_key_dim),
                     rmsbf2(I->gdn_v, C::gdn_value_dim));
        std::vector<std::uint16_t> hn(4);
        q.memcpy(hn.data(), I->h_norm, 4 * 2);
        q.wait();
        std::fprintf(stderr, "[verbose] GDN layer=%d h_norm[0:4]=%.4f,%.4f,%.4f,%.4f\n",
                     layer, from_bf16(hn[0]), from_bf16(hn[1]), from_bf16(hn[2]), from_bf16(hn[3]));
    }

    // 3. Gating: a = a_proj(h), b = b_proj(h); g, beta = gating(a, b)
    I->weights.gemv_bf16(base + "/a_projection", I->h_norm, I->gdn_a, q);
    I->weights.gemv_bf16(base + "/b_projection", I->h_norm, I->gdn_b, q);
    f32_to_bf16_(q, I->gdn_a, I->gdn_a_bf, 48);
    f32_to_bf16_(q, I->gdn_b, I->gdn_b_bf, 48);
    ops::gdn::gating(q, I->gdn_a_bf, I->gdn_b_bf, I->weights.fp32(base + "/a_log"),
                     I->weights.fp32(base + "/dt_bias"), I->gdn_g, I->gdn_beta);

    // 4. causal_conv1d_silu on [q,k,v]
    {
        // Concatenate q,k,v into gdn_qkv [10240].
        q.parallel_for(sycl::range<1>(C::gdn_key_dim), [=](sycl::id<1> id) {
            I->gdn_qkv[id[0]] = I->gdn_q[id[0]];
            I->gdn_qkv[C::gdn_key_dim + id[0]] = I->gdn_k[id[0]];
        });
        q.parallel_for(sycl::range<1>(C::gdn_value_dim), [=](sycl::id<1> id) {
            I->gdn_qkv[2 * C::gdn_key_dim + id[0]] = I->gdn_v[id[0]];
        });
        std::uint16_t* conv_state = I->gdn_conv + gidx * 3 * C::gdn_conv_dim;
        ops::gdn::causal_conv1d(q, I->gdn_qkv, I->weights.bf16(base + "/convolution"),
                                conv_state, I->gdn_qkv, C::gdn_conv_dim);
        // After conv: gdn_qkv[0:2048]=q_c, [2048:4096]=k_c, [4096:10240]=v_c
        q.memcpy(I->gdn_q, I->gdn_qkv, C::gdn_key_dim * 2);
        q.memcpy(I->gdn_k, I->gdn_qkv + C::gdn_key_dim, C::gdn_key_dim * 2);
        q.memcpy(I->gdn_v, I->gdn_qkv + 2 * C::gdn_key_dim, C::gdn_value_dim * 2);
    }

    // 5. normalize_qk + recurrent
    ops::gdn::normalize_qk(q, I->gdn_q, I->gdn_k);
    float* state = I->gdn_state + gidx * 48 * 128 * 128;
    ops::gdn::recurrent(q, I->gdn_q, I->gdn_k, I->gdn_v, I->gdn_g, I->gdn_beta, state,
                        I->gdn_out);

    // 6. gated_rmsnorm(o, gdn_norm, z)
    ops::gated_rmsnorm(q, I->gdn_out, I->weights.bf16(base + "/norm"),
                       I->gdn_vz_f + C::gdn_value_dim, C::rms_eps,
                       I->gdn_on, 48, 128);

    if (nsycl::verbose_enabled() && nsycl::log_layer(layer)) {
        // One-shot dump of GDN intermediates to the host.
        auto rms = [&](const float* p, int n) {
            std::vector<float> h(n);
            q.memcpy(h.data(), p, n * 4);
            q.wait();
            double ss = 0.0;
            for (float v : h) ss += (double)v * v;
            return std::sqrt(ss / n);
        };
        auto rmsbf = [&](const std::uint16_t* p, int n) {
            std::vector<std::uint16_t> h(n);
            q.memcpy(h.data(), p, n * 2);
            q.wait();
            double ss = 0.0;
            for (auto v : h) { float f = from_bf16(v); ss += (double)f * f; }
            return std::sqrt(ss / n);
        };
        std::vector<float> z(C::gdn_value_dim), out(C::gdn_value_dim), on(C::gdn_value_dim);
        std::vector<float> g(48), beta(48), a(48), b(48);
        q.memcpy(z.data(), I->gdn_vz_f + C::gdn_value_dim, C::gdn_value_dim * 4);
        q.memcpy(out.data(), I->gdn_out, C::gdn_value_dim * 4);
        q.memcpy(on.data(), I->gdn_on, C::gdn_value_dim * 4);
        q.memcpy(g.data(), I->gdn_g, 48 * 4);
        q.memcpy(beta.data(), I->gdn_beta, 48 * 4);
        q.memcpy(a.data(), I->gdn_a, 48 * 4);
        q.memcpy(b.data(), I->gdn_b, 48 * 4);
        q.wait();
        auto stats = [](const std::vector<float>& v) {
            double mn = 1e30, mx = -1e30, ss = 0;
            for (float x : v) { mn = std::min(mn, (double)x); mx = std::max(mx, (double)x); ss += (double)x * x; }
            return std::make_tuple(mn, mx, std::sqrt(ss / v.size()));
        };
        auto [zmn, zmx, zrms] = stats(z);
        auto [omn, omx, orms] = stats(out);
        auto [onmn, onmx, onrms] = stats(on);
        auto [amn, amx, arms] = stats(a);
        auto [bmn, bmx, brms] = stats(b);
        auto [gmn, gmx, grms] = stats(g);
        auto [betmn, betmx, betrms] = stats(beta);
        std::fprintf(stderr, "[verbose] GDN layer=%d a: min=%.4f max=%.4f rms=%.4f\n", layer, amn, amx, arms);
        std::fprintf(stderr, "[verbose] GDN layer=%d b: min=%.4f max=%.4f rms=%.4f\n", layer, bmn, bmx, brms);
        std::fprintf(stderr, "[verbose] GDN layer=%d g: min=%.4f max=%.4f rms=%.4f\n", layer, gmn, gmx, grms);
        std::fprintf(stderr, "[verbose] GDN layer=%d beta: min=%.4f max=%.4f rms=%.4f\n", layer, betmn, betmx, betrms);
        std::fprintf(stderr, "[verbose] GDN layer=%d z: min=%.4f max=%.4f rms=%.4f  z[0:4]=%.4f,%.4f,%.4f,%.4f\n",
                     layer, zmn, zmx, zrms, z[0], z[1], z[2], z[3]);
        std::fprintf(stderr, "[verbose] GDN layer=%d out: min=%.4f max=%.4f rms=%.4f\n", layer, omn, omx, orms);
        std::fprintf(stderr, "[verbose] GDN layer=%d on: min=%.4f max=%.4f rms=%.4f\n", layer, onmn, onmx, onrms);
        std::fprintf(stderr, "[verbose] GDN layer=%d q rms=%.4f k rms=%.4f v rms=%.4f\n", layer,
                     rmsbf(I->gdn_q, C::gdn_key_dim), rmsbf(I->gdn_k, C::gdn_key_dim),
                     rmsbf(I->gdn_v, C::gdn_value_dim));
    }

    // 7. x = x + output_proj(on)
    std::string outproj = "text/layers/" + std::to_string(layer) + "/gdn/output";
    f32_to_bf16_(q, I->gdn_on, I->gdn_out_bf, C::gdn_value_dim);
    I->weights.gemv(outproj, I->gdn_out_bf, I->mlp_down, q);
    q.parallel_for(sycl::range<1>(C::hidden), [=](sycl::id<1> id) {
        x[id[0]] = to_bf16(from_bf16(x[id[0]]) + I->mlp_down[id[0]]);
    });
}

void Model::mlp_layer(int layer, std::uint16_t* x, sycl::queue& q) {
    auto* I = impl_.get();
    std::string base = "text/layers/" + std::to_string(layer);

    // 1. h = rmsnorm(x, post_attention_norm)
    ops::rmsnorm(q, x, I->weights.bf16(base + "/post_attention_norm"), C::rms_eps,
                 I->h_norm, C::hidden);

    // 2. gate,up = gate_up_proj(h)
    I->weights.gemv(base + "/mlp/gate_up", I->h_norm, I->mlp_gate_up, q);

    // 3. h = silu(gate) * up
    ops::silu_mul(q, I->mlp_gate_up, I->mlp_gate_up + C::intermediate, I->mlp_act,
                  C::intermediate);

    // 4. x = x + down_proj(h)
    f32_to_bf16_(q, I->mlp_act, I->mlp_act_bf, C::intermediate);
    I->weights.gemv(base + "/mlp/down", I->mlp_act_bf, I->mlp_down, q);
    q.parallel_for(sycl::range<1>(C::hidden), [=](sycl::id<1> id) {
        x[id[0]] = to_bf16(from_bf16(x[id[0]]) + I->mlp_down[id[0]]);
    });
}

int Model::prefill(const std::vector<int>& tokens, sycl::queue& q) {
    auto* I = impl_.get();
    int tok_idx = 0;
    for (int tok : tokens) {
        I->weights.w8_row_dequant("text/token_embedding", tok, I->emb, q);
        q.memcpy(I->hidden, I->emb, C::hidden * 2);
        for (int layer = 0; layer < C::layers; ++layer) {
            if (C::is_full_attention(layer)) {
                attn_layer(C::full_attn_index(layer), I->hidden, q);
            } else {
                gdn_layer(C::gdn_index(layer), I->hidden, q);
            }
            mlp_layer(layer, I->hidden, q);
            if (nsycl::verbose_level() >= 3 && nsycl::log_layer(layer) && nsycl::log_token(tok_idx)) {
                std::vector<std::uint16_t> h(C::hidden);
                q.memcpy(h.data(), I->hidden, C::hidden * 2);
                q.wait();
                double ss = 0.0;
                for (auto v : h) { float f = from_bf16(v); ss += (double)f * f; }
                NINFER_VERBOSE("prefill tok=%d layer=%2d hidden rms=%.5f",
                               tok_idx, layer, std::sqrt(ss / C::hidden));
            }
        }
        last_token_ = tok;
        ++seq_len_;
        ++tok_idx;
    }
    // Compute the first generated token from the last prompt token's hidden state.
    int first = compute_logits(q);
    last_token_ = first;
    ++seq_len_;
    return first;
}

int Model::compute_logits(sycl::queue& q) {
    auto* I = impl_.get();
    // final_norm + output_head
    ops::rmsnorm(q, I->hidden, I->weights.bf16("text/final_norm"), C::rms_eps,
                 I->h_norm, C::hidden);
    I->weights.gemv("text/output_head", I->h_norm, I->logits, q);
    q.wait();
    std::vector<float> logits_host(C::vocab);
    q.memcpy(logits_host.data(), I->logits, C::vocab * 4);
    q.wait();
    int best = 0;
    float best_v = logits_host[0];
    for (int i = 1; i < C::token_domain; ++i) {
        if (logits_host[i] > best_v) { best_v = logits_host[i]; best = i; }
    }
    if (nsycl::verbose_enabled()) {
        // Top-5 logits for debugging (valid tokens only).
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < C::token_domain; ++i) top.emplace_back(logits_host[i], i);
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::fprintf(stderr, "[verbose] logits top5:");
        for (int i = 0; i < 5; ++i) std::fprintf(stderr, " (%d, %.3f)", top[i].second, top[i].first);
        std::fprintf(stderr, "\n");
    }
    return best;
}

int Model::decode(sycl::queue& q) {
    auto* I = impl_.get();
    // Embed the last generated token and run the full forward pass.
    I->weights.w8_row_dequant("text/token_embedding", last_token_, I->emb, q);
    q.memcpy(I->hidden, I->emb, C::hidden * 2);
    for (int layer = 0; layer < C::layers; ++layer) {
        if (C::is_full_attention(layer)) {
            attn_layer(C::full_attn_index(layer), I->hidden, q);
        } else {
            gdn_layer(C::gdn_index(layer), I->hidden, q);
        }
        mlp_layer(layer, I->hidden, q);
    }
    int best = compute_logits(q);
    last_token_ = best;
    ++seq_len_;
    return best;
}

} // namespace nsycl::model