// CPU double-precision reference forward pass for Qwen3.8-27B.
//
// Reads the same .ninfer artifact and runs the same math as model.cpp, but in
// double precision on the CPU. The residual stream is rounded to bf16 at each
// layer boundary (matching the GPU), so CPU and GPU should agree to ~f32
// precision. Any per-layer divergence pinpoints the buggy op.
//
// Usage: cpu_ref <artifact.ninfer> [tok tok ...]   (default tok 9707)

#include "artifact/reader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace nsycl::artifact;

// ---- config (mirrors ModelConfig) ----
namespace C {
constexpr int hidden       = 5120;
constexpr int layers       = 64;
constexpr int intermediate = 17408;
constexpr int vocab        = 248320;
constexpr int token_domain = 248077;  // rows [token_domain, vocab) are padding
constexpr int query_heads  = 24;
constexpr int kv_heads     = 4;
constexpr int head_dim     = 256;
constexpr int rotary_dim   = 64;
constexpr int query_size   = query_heads * head_dim; // 6144
constexpr int kv_size      = kv_heads * head_dim;    // 1024
constexpr int gdn_key_heads  = 16;
constexpr int gdn_value_heads = 48;
constexpr int gdn_head_dim   = 128;
constexpr int gdn_key_dim    = gdn_key_heads * gdn_head_dim;   // 2048
constexpr int gdn_value_dim  = gdn_value_heads * gdn_head_dim; // 6144
constexpr int gdn_conv_dim   = 2 * gdn_key_dim + gdn_value_dim; // 10240
constexpr int gdn_layers     = 48;
constexpr int full_attn_layers = 16;
constexpr float rms_eps     = 1.0e-6F;
constexpr float rope_theta  = 1.0e7F;
constexpr float kAttnScale  = 0.0625F;
constexpr float kGdnScale   = 0.08838834764831845F;
constexpr bool is_full_attention(int layer) { return (layer + 1) % 4 == 0; }
constexpr int full_attn_index(int layer) { return (layer + 1) / 4 - 1; }
constexpr int gdn_index(int layer) { return layer - (layer + 1) / 4; }
} // namespace C

// ---- bf16 helpers (match the GPU exactly) ----
static std::uint16_t to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
static float from_bf16(std::uint16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}
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

// ---- weight store: dequantize on demand into double rows ----
struct Tensor {
    std::vector<std::uint8_t> bytes; // raw payload
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
};

struct Store {
    Reader reader;
    std::unordered_map<std::string, Tensor> tensors;

    explicit Store(const std::filesystem::path& p) : reader(p) {
        for (const auto& obj : reader.objects()) {
            const auto name = std::string(object_name(obj));
            if (name.rfind("text/", 0) != 0) continue;
            if (const auto* t = std::get_if<TensorDescriptor>(&obj)) {
                Tensor tr;
                tr.shape  = t->shape;
                tr.format = t->format;
                tr.layout = t->layout;
                auto span = reader.payload(obj);
                tr.bytes.resize(span.data.size());
                std::memcpy(tr.bytes.data(), span.data.data(), span.data.size());
                tensors.emplace(name, std::move(tr));
            }
        }
    }

    const Tensor& get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("cpu_ref: missing " + name);
        return it->second;
    }

    // Dense bf16/f32 accessor.
    const std::uint16_t* dense_bf16(const std::string& name) const {
        const Tensor& t = get(name);
        return reinterpret_cast<const std::uint16_t*>(t.bytes.data());
    }
    const float* dense_f32(const std::string& name) const {
        const Tensor& t = get(name);
        return reinterpret_cast<const float*>(t.bytes.data());
    }

    // Quantized GEMV: out[n] = sum_k W[n,k]*x[k] in double.
    void gemv(const std::string& name, const std::vector<double>& x, std::vector<double>& out) const {
        const Tensor& t = get(name);
        const int N = (int)t.shape[0];
        const int K = (int)t.shape[1];
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        const std::uint8_t* low   = t.bytes.data();
        const std::uint8_t* high  = t.bytes.data() + geo.high_plane_offset;
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
        out.assign(N, 0.0);
#pragma omp parallel for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            const std::uint8_t* row_low   = low   + (std::size_t)n * gpr * 32;
            const std::uint8_t* row_high  = high  + (std::size_t)n * gpr * (t.format == NumericFormat::Q5G64_F16S ? 8 : 16);
            const std::uint16_t* row_scale = scale + (std::size_t)n * gpr;
            double acc = 0.0;
            for (int g = 0; g < gpr; ++g) {
                double s = f16_to_f32(row_scale[g]);
                const std::uint8_t* codes = row_low + g * 32;
                const std::uint8_t* hbits = row_high + g * (t.format == NumericFormat::Q5G64_F16S ? 8 : 16);
                double gs = 0.0;
                if (t.format == NumericFormat::Q4G64_F16S) {
                    for (int p = 0; p < 64; ++p) {
                        int k = g * 64 + p; if (k >= K) break;
                        std::uint8_t byte = codes[p >> 1];
                        int code = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                        int qv = (code ^ 8) - 8;
                        gs += (double)qv * x[k];
                    }
                } else if (t.format == NumericFormat::Q5G64_F16S) {
                    for (int p = 0; p < 64; ++p) {
                        int k = g * 64 + p; if (k >= K) break;
                        std::uint8_t byte = codes[p >> 1];
                        int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                        int hi1 = (hbits[p >> 3] >> (p & 7)) & 1;
                        int u = lo4 | (hi1 << 4);
                        int qv = (u ^ 16) - 16;
                        gs += (double)qv * x[k];
                    }
                } else if (t.format == NumericFormat::Q6G64_F16S) {
                    for (int p = 0; p < 64; ++p) {
                        int k = g * 64 + p; if (k >= K) break;
                        std::uint8_t byte = codes[p >> 1];
                        int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                        int hi2 = (hbits[p >> 1] >> ((p & 1) * 4)) & 0x03;
                        int u = lo4 | (hi2 << 4);
                        int qv = (u ^ 32) - 32;
                        gs += (double)qv * x[k];
                    }
                } else if (t.format == NumericFormat::W8G32_F16S) {
                    for (int p = 0; p < 32; ++p) {
                        int k = g * 32 + p; if (k >= K) break;
                        int qv = (int)(std::int8_t)codes[p];
                        gs += (double)qv * x[k];
                    }
                } else {
                    throw std::runtime_error("cpu_ref: unsupported format");
                }
                acc += s * gs;
            }
            out[n] = acc;
        }
    }

    // Dense BF16 GEMV: out[n] = sum_k W[n,k]*x[k] in double.
    void gemv_bf16(const std::string& name, const std::vector<double>& x, std::vector<double>& out) const {
        const Tensor& t = get(name);
        const int N = (int)t.shape[0];
        const int K = (int)t.shape[1];
        const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(t.bytes.data());
        out.assign(N, 0.0);
#pragma omp parallel for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            const std::uint16_t* row = w + (std::size_t)n * K;
            for (int k = 0; k < K; ++k) acc += (double)from_bf16(row[k]) * x[k];
            out[n] = acc;
        }
    }

    // W8 row dequant to bf16 (embedding).
    void w8_row(const std::string& name, int row, std::vector<std::uint16_t>& out) const {
        const Tensor& t = get(name);
        const int K = (int)t.shape[1];
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        const std::uint8_t* low   = t.bytes.data();
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
        out.assign(K, 0);
        for (int k = 0; k < K; ++k) {
            int g = k / 32, p = k % 32;
            const std::uint8_t* codes = low + (std::size_t)row * gpr * 32 + g * 32;
            float s = f16_to_f32(scale[(std::size_t)row * gpr + g]);
            int qv = (int)(std::int8_t)codes[p];
            out[k] = to_bf16((float)qv * s);
        }
    }
};

// ---- ops (double, bf16-rounded at the same boundaries as the GPU) ----
static std::vector<double> to_double(const std::vector<std::uint16_t>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(v[i]);
    return out;
}

static double rms(const std::vector<std::uint16_t>& v) {
    double ss = 0; for (auto b : v) { float f = from_bf16(b); ss += (double)f * f; }
    return std::sqrt(ss / v.size());
}

// rmsnorm: bf16 in/out.
static void rmsnorm(const std::vector<std::uint16_t>& x, const std::uint16_t* w,
                    std::vector<std::uint16_t>& out, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) { float v = from_bf16(x[i]); ss += (double)v * v; }
    double inv = 1.0 / std::sqrt(ss / n + C::rms_eps);
    out.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        // unit_offset: stored weight is (real - 1); effective = weight + 1.0.
        double v = (double)from_bf16(x[i]) * inv * ((double)from_bf16(w[i]) + 1.0);
        out[i] = to_bf16((float)v);
    }
}

// per-head rmsnorm: bf16 in/out.
static void rmsnorm_heads(const std::vector<std::uint16_t>& x, const std::uint16_t* w,
                          std::vector<std::uint16_t>& out, int heads, int hd) {
    out.assign(heads * hd, 0);
    for (int h = 0; h < heads; ++h) {
        double ss = 0;
        for (int i = 0; i < hd; ++i) { float v = from_bf16(x[(std::size_t)h * hd + i]); ss += (double)v * v; }
        double inv = 1.0 / std::sqrt(ss / hd + C::rms_eps);
        for (int i = 0; i < hd; ++i) {
            // unit_offset: stored weight is (real - 1); effective = weight + 1.0.
            double v = (double)from_bf16(x[(std::size_t)h * hd + i]) * inv * ((double)from_bf16(w[i]) + 1.0);
            out[(std::size_t)h * hd + i] = to_bf16((float)v);
        }
    }
}

// rope rotate-half on first `rot` dims. bf16 in/out.
static void rope(std::vector<std::uint16_t>& q, int qh, std::vector<std::uint16_t>& k, int kh,
                 int hd, int rot, int position) {
    auto apply = [&](std::vector<std::uint16_t>& ptr, int heads) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < rot / 2; ++i) {
                double freq = std::pow(C::rope_theta, -2.0 * i / rot);
                double ang  = position * freq;
                double s = std::sin(ang), c = std::cos(ang);
                double x0 = from_bf16(ptr[(std::size_t)h * hd + i]);
                double x1 = from_bf16(ptr[(std::size_t)h * hd + i + rot / 2]);
                ptr[(std::size_t)h * hd + i]         = to_bf16((float)(x0 * c - x1 * s));
                ptr[(std::size_t)h * hd + i + rot / 2] = to_bf16((float)(x1 * c + x0 * s));
            }
        }
    };
    apply(q, qh);
    apply(k, kh);
}

// gqa decode (single token). q,k,v bf16; out double [qh*hd].
static void gqa_decode(const std::vector<std::uint16_t>& q, const std::vector<std::uint16_t>& k_cur,
                       const std::vector<std::uint16_t>& v_cur, const std::vector<std::uint16_t>& cache_k,
                       const std::vector<std::uint16_t>& cache_v, int seq_len, std::vector<double>& out) {
    const int qh = C::query_heads, kvh = C::kv_heads, hd = C::head_dim, grp = qh / kvh;
    out.assign(qh * hd, 0.0);
#pragma omp parallel for schedule(dynamic, 2)
    for (int hq = 0; hq < qh; ++hq) {
        int kv = hq / grp;
        double m = -1e30, l = 0.0;
        std::vector<double> acc(hd, 0.0);
        for (int t = 0; t < seq_len; ++t) {
            const std::uint16_t* krow; const std::uint16_t* vrow;
            if (t == seq_len - 1) {
                krow = k_cur.data() + (std::size_t)kv * hd;
                vrow = v_cur.data() + (std::size_t)kv * hd;
            } else {
                krow = cache_k.data() + ((std::size_t)t * kvh + kv) * hd;
                vrow = cache_v.data() + ((std::size_t)t * kvh + kv) * hd;
            }
            double s = 0;
            for (int d = 0; d < hd; ++d) s += (double)from_bf16(q[(std::size_t)hq * hd + d]) * from_bf16(krow[d]);
            s *= C::kAttnScale;
            double m_new = std::max(s, m);
            double scale = std::exp(m - m_new);
            l = l * scale + std::exp(s - m_new);
            for (int d = 0; d < hd; ++d) acc[d] = acc[d] * scale + std::exp(s - m_new) * from_bf16(vrow[d]);
            m = m_new;
        }
        for (int d = 0; d < hd; ++d) out[(std::size_t)hq * hd + d] = acc[d] / l;
    }
}

// gated_rmsnorm: o,z double; weight bf16; out double. gate = SiLU(z).
static void gated_rmsnorm(const std::vector<double>& o, const std::uint16_t* w,
                          const std::vector<double>& z, std::vector<double>& out, int heads, int dim) {
    out.assign(heads * dim, 0.0);
    for (int h = 0; h < heads; ++h) {
        double ss = 0;
        for (int i = 0; i < dim; ++i) ss += o[(std::size_t)h * dim + i] * o[(std::size_t)h * dim + i];
        double inv = 1.0 / std::sqrt(ss / dim + C::rms_eps);
        for (int i = 0; i < dim; ++i) {
            double wi = from_bf16(w[i]);
            double zi = z[(std::size_t)h * dim + i];
            double silu = zi / (1.0 + std::exp(-zi));
            out[(std::size_t)h * dim + i] = o[(std::size_t)h * dim + i] * inv * wi * silu;
        }
    }
}

// causal_conv1d k=4 + SiLU, single token. x bf16 [C], weight bf16 [4,C],
// state bf16 [3,C] (row0 oldest). out bf16 [C]. state updated in place.
static void causal_conv1d(const std::vector<std::uint16_t>& x, const std::uint16_t* weight,
                          std::vector<std::uint16_t>& state, std::vector<std::uint16_t>& out, int Cn) {
    out.assign(Cn, 0);
    for (int c = 0; c < Cn; ++c) {
        double s0 = from_bf16(state[c]);
        double s1 = from_bf16(state[Cn + c]);
        double s2 = from_bf16(state[2 * Cn + c]);
        double w0 = from_bf16(weight[c]);
        double w1 = from_bf16(weight[Cn + c]);
        double w2 = from_bf16(weight[2 * Cn + c]);
        double w3 = from_bf16(weight[3 * Cn + c]);
        double x0 = from_bf16(x[c]);
        double acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x0;
        double silu = acc / (1.0 + std::exp(-acc));
        out[c] = to_bf16((float)silu);
        state[c]         = to_bf16((float)s1);
        state[Cn + c]    = to_bf16((float)s2);
        state[2 * Cn + c] = to_bf16((float)x0);
    }
}

// L2-normalize q,k per head (16 heads x 128). bf16 in/out.
static void normalize_qk(std::vector<std::uint16_t>& q, std::vector<std::uint16_t>& k) {
    auto norm = [&](std::vector<std::uint16_t>& ptr) {
        for (int h = 0; h < C::gdn_key_heads; ++h) {
            double ss = 0;
            std::vector<double> vals(C::gdn_head_dim);
            for (int i = 0; i < C::gdn_head_dim; ++i) {
                double v = from_bf16(ptr[(std::size_t)h * C::gdn_head_dim + i]);
                vals[i] = v; ss += v * v;
            }
            double inv = 1.0 / std::sqrt(ss + 1e-6);
            for (int i = 0; i < C::gdn_head_dim; ++i)
                ptr[(std::size_t)h * C::gdn_head_dim + i] = to_bf16((float)(vals[i] * inv));
        }
    };
    norm(q); norm(k);
}

// GDN recurrent decode (single token). q,k,v bf16; g,beta double [48];
// state double [48][128][128]; out double [48*128].
static void gdn_recurrent(const std::vector<std::uint16_t>& q, const std::vector<std::uint16_t>& k,
                          const std::vector<std::uint16_t>& v, const std::vector<double>& g,
                          const std::vector<double>& beta, double* state,
                          std::vector<double>& out) {
    const int HV = C::gdn_value_heads, HD = C::gdn_head_dim;
    out.assign(HV * HD, 0.0);
#pragma omp parallel for schedule(dynamic, 4)
    for (int hv = 0; hv < HV; ++hv) {
        int qk = hv / 3;
        double alpha = std::exp(g[hv]);
        double beta_v = beta[hv];
        std::vector<double> kv(HD), qv(HD);
        for (int c = 0; c < HD; ++c) kv[c] = from_bf16(k[(std::size_t)qk * HD + c]);
        for (int r = 0; r < HD; ++r) {
            double* st = state + ((std::size_t)hv * HD + r) * HD;
            double partial = 0;
            for (int c = 0; c < HD; ++c) partial += st[c] * kv[c];
            double v_r = from_bf16(v[(std::size_t)hv * HD + r]);
            double delta = beta_v * (v_r - alpha * partial);
            for (int c = 0; c < HD; ++c) st[c] = alpha * st[c] + delta * kv[c];
        }
        for (int c = 0; c < HD; ++c) qv[c] = from_bf16(q[(std::size_t)qk * HD + c]);
        for (int r = 0; r < HD; ++r) {
            const double* st = state + ((std::size_t)hv * HD + r) * HD;
            double attn = 0;
            for (int c = 0; c < HD; ++c) attn += st[c] * qv[c];
            out[(std::size_t)hv * HD + r] = attn * C::kGdnScale;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact.ninfer> [tok ...]\n", argv[0]); return 1; }
    std::vector<int> prompt;
    for (int i = 2; i < argc; ++i) prompt.push_back(std::atoi(argv[i]));
    if (prompt.empty()) prompt = {9707};

    Store store(argv[1]);

    // Persistent state.
    // Per-layer KV caches: 16 full-attention layers, each token-major [seq, kvh, hd].
    // Preallocated to the full prompt length (decode appends one token at a time).
    const int n_full = C::full_attn_layers;
    std::vector<std::vector<std::uint16_t>> cache_k(n_full), cache_v(n_full);
    for (int f = 0; f < n_full; ++f) {
        cache_k[f].resize((std::size_t)prompt.size() * C::kv_size);
        cache_v[f].resize((std::size_t)prompt.size() * C::kv_size);
    }
    std::vector<double> gdn_state((std::size_t)C::gdn_layers * 48 * 128 * 128, 0.0);
    std::vector<std::uint16_t> gdn_conv((std::size_t)C::gdn_layers * 3 * C::gdn_conv_dim, 0);

    std::vector<std::uint16_t> x; // residual stream (bf16)
    int seq_len = 0;

    for (int tok : prompt) {
        std::vector<std::uint16_t> emb;
        store.w8_row("text/token_embedding", tok, emb);
        x = emb;
        // Per-token embedding diagnostic: is the embedding itself the outlier source?
        {
            double ss = 0, mx = -1e30; int mxp = 0;
            for (int i = 0; i < C::hidden; ++i) { double f = from_bf16(emb[i]); ss += f*f; if (f > mx) { mx = f; mxp = i; } }
            std::fprintf(stderr, "[emb] tok=%2d id=%d rms=%.4f max=%.3f@%d x3994=%.3f x3456=%.3f\n",
                         seq_len, tok, std::sqrt(ss / C::hidden), mx, mxp, from_bf16(emb[3994]), from_bf16(emb[3456]));
        }
        if (seq_len == (int)prompt.size() - 1)
            std::fprintf(stderr, "[trace] tok=%d EMB x[3994]=%.3f x[0]=%.3f\n", seq_len, from_bf16(x[3994]), from_bf16(x[0]));
        for (int layer = 0; layer < C::layers; ++layer) {
            if (C::is_full_attention(layer)) {
                int fidx = C::full_attn_index(layer);
                std::string base = "text/layers/" + std::to_string(4 * (fidx + 1) - 1);
                std::vector<std::uint16_t> h;
                rmsnorm(x, store.dense_bf16(base + "/input_norm"), h, C::hidden);
                std::vector<double> qkv, gv;
                store.gemv(base + "/attention/query_key", to_double(h), qkv);
                store.gemv(base + "/attention/gate_value", to_double(h), gv);
                std::vector<std::uint16_t> q(C::query_size), k(C::kv_size), gate(C::query_size), v(C::kv_size);
                for (int i = 0; i < C::query_size; ++i) q[i] = to_bf16((float)qkv[i]);
                for (int i = 0; i < C::kv_size; ++i) k[i] = to_bf16((float)qkv[C::query_size + i]);
                for (int i = 0; i < C::query_size; ++i) gate[i] = to_bf16((float)gv[i]);
                for (int i = 0; i < C::kv_size; ++i) v[i] = to_bf16((float)gv[C::query_size + i]);
                std::vector<std::uint16_t> qn, kn;
                rmsnorm_heads(q, store.dense_bf16(base + "/attention/query_norm"), qn, C::query_heads, C::head_dim);
                rmsnorm_heads(k, store.dense_bf16(base + "/attention/key_norm"), kn, C::kv_heads, C::head_dim);
                rope(qn, C::query_heads, kn, C::kv_heads, C::head_dim, C::rotary_dim, seq_len);
                // write current k/v into this layer's cache at seq_len
                {
                    std::size_t off = (std::size_t)seq_len * C::kv_size;
                    for (int i = 0; i < C::kv_size; ++i) { cache_k[fidx][off + i] = kn[i]; cache_v[fidx][off + i] = v[i]; }
                }
                std::vector<double> attn;
                gqa_decode(qn, kn, v, cache_k[fidx], cache_v[fidx], seq_len + 1, attn);
                // Deep dump (matches GPU [deep] format): head-0 scores/softmax +
                // gated attn_out, recomputed independently from qn + cache.
                if (seq_len == (int)prompt.size() - 1) {
                    const int layer = 4 * (fidx + 1) - 1;
                    const int S = seq_len + 1;
                    const int hd = C::head_dim, kvh = C::kv_heads;
                    std::vector<double> qd(hd);
                    for (int d = 0; d < hd; ++d) qd[d] = from_bf16(qn[d]);
                    std::vector<double> scores(S);
                    double mx = -1e30;
                    for (int t = 0; t < S; ++t) {
                        double s = 0.0;
                        for (int d = 0; d < hd; ++d)
                            s += qd[d] * from_bf16(cache_k[fidx][std::size_t(t) * kvh * hd + d]);
                        s *= C::kAttnScale;
                        scores[t] = s;
                        if (s > mx) mx = s;
                    }
                    double l = 0.0;
                    std::vector<double> w(S);
                    for (int t = 0; t < S; ++t) { w[t] = std::exp(scores[t] - mx); l += w[t]; }
                    for (int t = 0; t < S; ++t) w[t] /= l;
                    std::vector<double> host_aout(hd, 0.0);
                    for (int d = 0; d < hd; ++d)
                        for (int t = 0; t < S; ++t)
                            host_aout[d] += w[t] * from_bf16(cache_v[fidx][std::size_t(t) * kvh * hd + d]);
                    // Apply the gate to the host recompute (matches the gated
                    // kernel output).
                    std::vector<double> host_gated(hd);
                    double gmin = 1e30, gmax = -1e30, gss = 0;
                    for (int d = 0; d < hd; ++d) {
                        double g = from_bf16(gate[d]);
                        double sg = 1.0 / (1.0 + std::exp(-g));
                        host_gated[d] = sg * host_aout[d];
                        gmin = std::min(gmin, sg); gmax = std::max(gmax, sg); gss += sg * sg;
                    }
                    double maxdiff = 0.0;
                    for (int d = 0; d < hd; ++d)
                        maxdiff = std::max(maxdiff, std::abs(attn[d] - host_gated[d]));
                    std::fprintf(stderr, "[deep] ATTN layer=%d pos=%d S=%d\n", layer, seq_len, S);
                    std::fprintf(stderr, "[deep]   qn0[0:8]=");
                    for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.4f ", from_bf16(qn[d]));
                    std::fprintf(stderr, "\n[deep]   scores[0:%d]=", S);
                    for (int t = 0; t < S; ++t) std::fprintf(stderr, "%.4f ", scores[t]);
                    std::fprintf(stderr, "\n[deep]   softmax[0:%d]=", S);
                    for (int t = 0; t < S; ++t) std::fprintf(stderr, "%.5f ", w[t]);
                    std::fprintf(stderr, "\n[deep]   gate_sigmoid head0: min=%.4f max=%.4f rms=%.4f\n",
                                 gmin, gmax, std::sqrt(gss / hd));
                    std::fprintf(stderr, "[deep]   kernel aout0[0:8]=");
                    for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.5f ", attn[d]);
                    std::fprintf(stderr, "\n[deep]   host   aout0[0:8]=");
                    for (int d = 0; d < 8; ++d) std::fprintf(stderr, "%.5f ", host_gated[d]);
                    std::fprintf(stderr, "\n[deep]   kernel-vs-host maxdiff=%.6f\n", maxdiff);
                }
                for (int i = 0; i < C::query_size; ++i)
                    attn[i] = (1.0 / (1.0 + std::exp(-(double)from_bf16(gate[i]))) * attn[i]);
                {
                    auto rmsbf = [](const std::vector<std::uint16_t>& v) {
                        double ss = 0; for (auto b : v) { float f = from_bf16(b); ss += (double)f * f; }
                        return std::sqrt(ss / v.size());
                    };
                    double ss = 0; for (double v : attn) ss += v * v;
                    std::fprintf(stderr, "[cpu] ATTN layer=%d qn rms=%.4f kn rms=%.4f v rms=%.4f gate rms=%.4f attn_out rms=%.4f\n",
                                 4 * (fidx + 1) - 1, rmsbf(qn), rmsbf(kn), rmsbf(v), rmsbf(gate), std::sqrt(ss / attn.size()));
                }
                std::vector<std::uint16_t> attn_bf(C::query_size);
                for (int i = 0; i < C::query_size; ++i) attn_bf[i] = to_bf16((float)attn[i]);
                std::vector<double> down;
                store.gemv(base + "/attention/output", to_double(attn_bf), down);
                for (int i = 0; i < C::hidden; ++i)
                    x[i] = to_bf16(from_bf16(x[i]) + (float)down[i]);
            } else {
                int gidx = C::gdn_index(layer);
                int lyr = 4 * (gidx / 3) + (gidx % 3);
                std::string base = "text/layers/" + std::to_string(lyr) + "/gdn";
                std::vector<std::uint16_t> h;
                rmsnorm(x, store.dense_bf16("text/layers/" + std::to_string(lyr) + "/input_norm"), h, C::hidden);
                std::vector<double> qkv, vz;
                store.gemv(base + "/query_key", to_double(h), qkv);
                store.gemv(base + "/value_z", to_double(h), vz);
                std::vector<std::uint16_t> q(C::gdn_key_dim), k(C::gdn_key_dim), v(C::gdn_value_dim);
                for (int i = 0; i < C::gdn_key_dim; ++i) q[i] = to_bf16((float)qkv[i]);
                for (int i = 0; i < C::gdn_key_dim; ++i) k[i] = to_bf16((float)qkv[C::gdn_key_dim + i]);
                for (int i = 0; i < C::gdn_value_dim; ++i) v[i] = to_bf16((float)vz[i]);
                std::vector<double> z(C::gdn_value_dim);
                for (int i = 0; i < C::gdn_value_dim; ++i) z[i] = vz[C::gdn_value_dim + i];
                // gating
                std::vector<double> a, b;
                store.gemv_bf16(base + "/a_projection", to_double(h), a);
                store.gemv_bf16(base + "/b_projection", to_double(h), b);
                std::vector<double> g(48), beta(48);
                const float* a_log = store.dense_f32(base + "/a_log");
                const float* dt_bias = store.dense_f32(base + "/dt_bias");
                for (int hh = 0; hh < 48; ++hh) {
                    double sp = a[hh] + dt_bias[hh];
                    sp = (sp > 20.0) ? sp : std::log1p(std::exp(sp));
                    g[hh] = -std::exp(a_log[hh]) * sp;
                    beta[hh] = 1.0 / (1.0 + std::exp(-b[hh]));
                }
                // causal conv on [q,k,v]
                std::vector<std::uint16_t> qkv_cat(C::gdn_conv_dim);
                for (int i = 0; i < C::gdn_key_dim; ++i) qkv_cat[i] = q[i];
                for (int i = 0; i < C::gdn_key_dim; ++i) qkv_cat[C::gdn_key_dim + i] = k[i];
                for (int i = 0; i < C::gdn_value_dim; ++i) qkv_cat[2 * C::gdn_key_dim + i] = v[i];
                std::vector<std::uint16_t> conv_state(3 * C::gdn_conv_dim);
                std::copy(gdn_conv.begin() + (std::size_t)gidx * 3 * C::gdn_conv_dim,
                          gdn_conv.begin() + (std::size_t)(gidx + 1) * 3 * C::gdn_conv_dim, conv_state.begin());
                std::vector<std::uint16_t> qkv_c;
                causal_conv1d(qkv_cat, store.dense_bf16(base + "/convolution"), conv_state, qkv_c, C::gdn_conv_dim);
                std::copy(conv_state.begin(), conv_state.end(),
                          gdn_conv.begin() + (std::size_t)gidx * 3 * C::gdn_conv_dim);
                for (int i = 0; i < C::gdn_key_dim; ++i) q[i] = qkv_c[i];
                for (int i = 0; i < C::gdn_key_dim; ++i) k[i] = qkv_c[C::gdn_key_dim + i];
                for (int i = 0; i < C::gdn_value_dim; ++i) v[i] = qkv_c[2 * C::gdn_key_dim + i];
                normalize_qk(q, k);
                std::vector<double> out;
                gdn_recurrent(q, k, v, g, beta,
                              gdn_state.data() + (std::size_t)gidx * 48 * 128 * 128, out);
                std::vector<double> on;
                gated_rmsnorm(out, store.dense_bf16(base + "/norm"), z, on, 48, 128);
                // Per-token GDN layer-0 trajectory: does the recurrent state blow up?
                if (layer == 0) {
                    double orss = 0, omx = -1e30;
                    for (double d : out) { orss += d*d; omx = std::max(omx, std::abs(d)); }
                    double ssrs = 0;
                    for (int i = 0; i < 48 * 128 * 128; ++i) { double v = gdn_state[(std::size_t)i]; ssrs += v*v; }
                    std::fprintf(stderr, "[gdntraj] tok=%2d o_rms=%.4f o_max=%.3f state_rms=%.4f\n",
                                 seq_len, std::sqrt(orss / out.size()), omx, std::sqrt(ssrs / (48 * 128 * 128)));
                }
                if (seq_len == (int)prompt.size() - 1 && layer == 0) {
                    double o_rms = 0, on_rms = 0;
                    for (double v : out) o_rms += v*v;
                    for (double v : on) on_rms += v*v;
                    o_rms = std::sqrt(o_rms / out.size());
                    on_rms = std::sqrt(on_rms / on.size());
                    std::fprintf(stderr, "[gdn0] o rms=%.6f | on rms=%.6f | on max=%.4f\n", o_rms, on_rms, *std::max_element(on.begin(), on.end()));
                }
                std::vector<std::uint16_t> on_bf(C::gdn_value_dim);
                for (int i = 0; i < C::gdn_value_dim; ++i) on_bf[i] = to_bf16((float)on[i]);
                std::vector<double> down;
                store.gemv(base + "/output", to_double(on_bf), down);
                // Per-token GDN down-proj at 3994: resolve the emb->x3994 jump.
                if (layer == 0) {
                    double onss = 0, onmx = -1e30;
                    for (double d : on) { onss += d*d; onmx = std::max(onmx, std::abs(d)); }
                    std::fprintf(stderr, "[gdn3994] tok=%2d gdn_down3994=%.3f on_rms=%.4f on_max=%.3f\n",
                                 seq_len, down[3994], std::sqrt(onss / on.size()), onmx);
                }
                for (int i = 0; i < C::hidden; ++i)
                    x[i] = to_bf16(from_bf16(x[i]) + (float)down[i]);
                if (seq_len == (int)prompt.size() - 1 && layer == 0) {
                    auto dmax = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
                    auto dmin = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
                    std::fprintf(stderr, "[gdn0] z max=%.3f min=%.3f | on max=%.3f min=%.3f | down max=%.3f min=%.3f\n",
                                 dmax(z), dmin(z), dmax(on), dmin(on), dmax(down), dmin(down));
                    // Which input position drives down[3994]?
                    // down[3994] = sum_k W[3994,k] * on_bf[k]
                    // Find the top contributor.
                    const auto& t = store.get(base + "/output");
                    const int N = (int)t.shape[0], K = (int)t.shape[1];
                    const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
                    auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
                    const int gpr = (int)geo.groups_per_row;
                    const std::uint8_t* low   = t.bytes.data();
                    const std::uint8_t* high  = t.bytes.data() + geo.high_plane_offset;
                    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
                    const std::uint8_t* row_low   = low + (std::size_t)3994 * gpr * 32;
                    const std::uint8_t* row_high  = high + (std::size_t)3994 * gpr * 8;
                    const std::uint16_t* row_scale = scale + (std::size_t)3994 * gpr;
                    double ss = 0; double mn = 1e30, mx = -1e30;
                    for (int g = 0; g < gpr; ++g) {
                        double s = f16_to_f32(row_scale[g]);
                        for (int p = 0; p < 64; ++p) {
                            int k = g * 64 + p; if (k >= K) break;
                            std::uint8_t byte = row_low[(g * 64 + p) >> 1];
                            int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                            int hi1 = (row_high[(g * 64 + p) >> 3] >> (p & 7)) & 1;
                            int u = lo4 | (hi1 << 4);
                            int qv = (u ^ 16) - 16;
                            double w = (double)qv * s;
                            ss += w * w; mn = std::min(mn, w); mx = std::max(mx, w);
                        }
                    }
                    std::fprintf(stderr, "[gdn0] output_proj row 3994: rms=%.5f min=%.4f max=%.4f\n",
                                 std::sqrt(ss / K), mn, mx);
                }
            }
            // MLP
            std::string mbase = "text/layers/" + std::to_string(layer);
            std::vector<std::uint16_t> h;
            rmsnorm(x, store.dense_bf16(mbase + "/post_attention_norm"), h, C::hidden);
            std::vector<double> gu;
            store.gemv(mbase + "/mlp/gate_up", to_double(h), gu);
            std::vector<double> act(C::intermediate);
            for (int i = 0; i < C::intermediate; ++i)
                act[i] = (gu[i] / (1.0 + std::exp(-gu[i]))) * gu[C::intermediate + i];
            std::vector<std::uint16_t> act_bf(C::intermediate);
            for (int i = 0; i < C::intermediate; ++i) act_bf[i] = to_bf16((float)act[i]);
            std::vector<double> down;
            store.gemv(mbase + "/mlp/down", to_double(act_bf), down);
            // Per-token layer-0 trajectory: when does the x3994 outlier appear?
            if (layer == 0) {
                double arss = 0; double amx = -1e30;
                for (double d : act) { arss += d*d; amx = std::max(amx, d); }
                std::fprintf(stderr, "[traj] tok=%2d x3994=%.3f mlp_down3994=%.3f act_rms=%.4f act_max=%.3f\n",
                             seq_len, (double)from_bf16(x[3994]), down[3994], std::sqrt(arss / act.size()), amx);
            }
            if (seq_len == (int)prompt.size() - 1 && layer == 0) {
                double act_max = *std::max_element(act.begin(), act.end());
                double act_min = *std::min_element(act.begin(), act.end());
                std::fprintf(stderr, "[iso] layer0 x3994 before MLP=%.3f | mlp_down[3994]=%.3f | act rms=%.4f max=%.4f min=%.4f\n",
                             (double)from_bf16(x[3994]), down[3994],
                             [](const std::vector<double>& v){double ss=0;for(double d:v)ss+=d*d;return std::sqrt(ss/v.size());}(act), act_max, act_min);
                // Check MLP down row 3994 rms using the same dequant as gemv
                {
                    const auto& t = store.get(mbase + "/mlp/down");
                    const int N = (int)t.shape[0], K = (int)t.shape[1];
                    const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
                    auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
                    const int gpr = (int)geo.groups_per_row;
                    const std::uint8_t* low   = t.bytes.data();
                    const std::uint8_t* high  = t.bytes.data() + geo.high_plane_offset;
                    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
                    // Determine format: Q5G64 or Q4G64
                    bool is_q5 = (t.format == NumericFormat::Q5G64_F16S);
                    bool is_q4 = (t.format == NumericFormat::Q4G64_F16S);
                    double ss = 0; double mn = 1e30, mx = -1e30; int cnt = 0;
                    for (int g = 0; g < gpr; ++g) {
                        double s = f16_to_f32(scale[(std::size_t)3994 * gpr + g]);
                        for (int p = 0; p < 64; ++p) {
                            int k = g * 64 + p; if (k >= K) break;
                            int qv;
                            if (is_q5) {
                                std::uint8_t byte = low[((std::size_t)3994 * gpr + g) * 32 + (p >> 1)];
                                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                                int hi1 = (high[((std::size_t)3994 * gpr + g) * 8 + (p >> 3)] >> (p & 7)) & 1;
                                int u = lo4 | (hi1 << 4);
                                qv = (u ^ 16) - 16;
                            } else {
                                std::uint8_t byte = low[((std::size_t)3994 * gpr + g) * 32 + (p >> 1)];
                                qv = ((p & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;
                            }
                            double w = (double)qv * s;
                            ss += w * w; cnt++;
                            mn = std::min(mn, w); mx = std::max(mx, w);
                        }
                    }
                    std::fprintf(stderr, "[iso] mlp/down row 3994: fmt=%s rms=%.5f min=%.4f max=%.4f\n",
                                 is_q5 ? "Q5" : "Q4", std::sqrt(ss / cnt), mn, mx);
                    // Also check row 0 for comparison
                    ss = 0; mn = 1e30; mx = -1e30; cnt = 0;
                    for (int g = 0; g < gpr; ++g) {
                        double s = f16_to_f32(scale[(std::size_t)0 * gpr + g]);
                        for (int p = 0; p < 64; ++p) {
                            int k = g * 64 + p; if (k >= K) break;
                            int qv;
                            if (is_q5) {
                                std::uint8_t byte = low[((std::size_t)0 * gpr + g) * 32 + (p >> 1)];
                                int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                                int hi1 = (high[((std::size_t)0 * gpr + g) * 8 + (p >> 3)] >> (p & 7)) & 1;
                                int u = lo4 | (hi1 << 4);
                                qv = (u ^ 16) - 16;
                            } else {
                                std::uint8_t byte = low[((std::size_t)0 * gpr + g) * 32 + (p >> 1)];
                                qv = ((p & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;
                            }
                            double w = (double)qv * s;
                            ss += w * w; cnt++;
                            mn = std::min(mn, w); mx = std::max(mx, w);
                        }
                    }
                    std::fprintf(stderr, "[iso] mlp/down row 0:      fmt=%s rms=%.5f min=%.4f max=%.4f\n",
                                 is_q5 ? "Q5" : "Q4", std::sqrt(ss / cnt), mn, mx);
                }
                // ---- THOROUGH MLP-PATH DIAGNOSTICS (layer 0, last token) ----
                auto dstat = [](const std::vector<double>& v, const char* tag) {
                    double s2 = 0, mn2 = 1e30, mx2 = -1e30; int mxp = 0;
                    for (int i = 0; i < (int)v.size(); ++i) { double f = v[i]; s2 += f*f; if (f > mx2) { mx2 = f; mxp = i; } mn2 = std::min(mn2, f); }
                    std::fprintf(stderr, "[mlp] %s rms=%.5f max=%.4f@%d min=%.4f\n", tag, std::sqrt(s2 / v.size()), mx2, mxp, mn2);
                };
                std::vector<double> h_d(C::hidden);
                for (int i = 0; i < C::hidden; ++i) h_d[i] = from_bf16(h[i]);
                dstat(h_d, "h_norm");
                std::vector<double> gate(C::intermediate), up(C::intermediate);
                for (int i = 0; i < C::intermediate; ++i) { gate[i] = gu[i]; up[i] = gu[C::intermediate + i]; }
                dstat(gate, "gate");
                dstat(up,   "up");
                dstat(act,  "act");
                {
                    std::vector<std::pair<double,int>> av;
                    for (int i = 0; i < C::intermediate; ++i) av.emplace_back(std::abs(act[i]), i);
                    std::partial_sort(av.begin(), av.begin() + 10, av.end(),
                                      [](const auto& a, const auto& b){ return a.first > b.first; });
                    std::fprintf(stderr, "[mlp] act top10 |v|:");
                    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " (%d, %.3f)", av[i].second, act[av[i].second]);
                    std::fprintf(stderr, "\n");
                }
                {
                    const auto& t = store.get(mbase + "/mlp/down");
                    const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
                    auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
                    const int gpr = (int)geo.groups_per_row;
                    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
                    for (int row : {3994, 0}) {
                        double smn = 1e30, smx = -1e30, sss = 0; int sn = 0;
                        for (int g = 0; g < gpr; ++g) {
                            float s = f16_to_f32(scale[(std::size_t)row * gpr + g]);
                            smn = std::min(smn, (double)s); smx = std::max(smx, (double)s); sss += (double)s; sn++;
                        }
                        std::fprintf(stderr, "[mlp] down row %d scale dist: min=%.6f max=%.6f mean=%.6f (n=%d)\n", row, smn, smx, sss / sn, sn);
                        std::fprintf(stderr, "[mlp] down row %d raw scales[0:16]:", row);
                        for (int g = 0; g < 16 && g < gpr; ++g) {
                            std::uint16_t sb = scale[(std::size_t)row * gpr + g];
                            std::fprintf(stderr, " %04x=%.5f", sb, f16_to_f32(sb));
                        }
                        std::fprintf(stderr, "\n");
                    }
                }
                {
                    const auto& t = store.get(mbase + "/mlp/down");
                    const int K = (int)t.shape[1];
                    const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
                    auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
                    const int gpr = (int)geo.groups_per_row;
                    const std::uint8_t* low   = t.bytes.data();
                    const std::uint8_t* high  = t.bytes.data() + geo.high_plane_offset;
                    const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
                    std::vector<double> wrow(K, 0.0);
                    for (int g = 0; g < gpr; ++g) {
                        double s = f16_to_f32(scale[(std::size_t)3994 * gpr + g]);
                        const std::uint8_t* codes = low + (std::size_t)3994 * gpr * 32 + g * 32;
                        const std::uint8_t* hbits = high + (std::size_t)3994 * gpr * 8 + g * 8;
                        for (int p = 0; p < 64; ++p) {
                            int k = g * 64 + p; if (k >= K) break;
                            std::uint8_t byte = codes[p >> 1];
                            int lo4 = (p & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
                            int hi1 = (hbits[p >> 3] >> (p & 7)) & 1;
                            int u = lo4 | (hi1 << 4);
                            int qv = (u ^ 16) - 16;
                            wrow[k] = (double)qv * s;
                        }
                    }
                    double dot = 0, wss = 0, ass = 0;
                    for (int k = 0; k < K; ++k) { dot += wrow[k] * act[k]; wss += wrow[k]*wrow[k]; ass += act[k]*act[k]; }
                    double corr = dot / (std::sqrt(wss) * std::sqrt(ass) + 1e-30);
                    std::fprintf(stderr, "[mlp] act·down[3994] dot=%.3f corr=%.4f (gemv down[3994]=%.3f)\n", dot, corr, down[3994]);
                }
            }
            for (int i = 0; i < C::hidden; ++i)
                x[i] = to_bf16(from_bf16(x[i]) + (float)down[i]);
            {
                double mn = 1e30, mx = -1e30; int mxpos = 0;
                for (int i = 0; i < C::hidden; ++i) { double f = from_bf16(x[i]); if (f > mx) { mx = f; mxpos = i; } mn = std::min(mn, f); }
                // count elements with |v| > 10
                int big = 0; for (auto b : x) if (std::abs(from_bf16(b)) > 10.0) big++;
                std::fprintf(stderr, "[cpu] tok=%d layer=%2d hidden rms=%.4f max=%.2f@%d min=%.2f big(>10)=%d x3994=%.3f\n",
                             seq_len, layer, rms(x), mx, mxpos, mn, big, from_bf16(x[3994]));
            }
        }
        ++seq_len;
    }

    // GDN state diagnostic: check magnitude of the recurrent state after prefill.
    {
        double ss = 0.0, mn = 1e30, mx = -1e30;
        for (double v : gdn_state) { ss += v * v; mn = std::min(mn, v); mx = std::max(mx, v); }
        std::fprintf(stderr, "[diag] GDN state (all 48 layers): rms=%.5f min=%.4f max=%.4f\n",
                     std::sqrt(ss / gdn_state.size()), mn, mx);
        // Per-layer state rms for the first few GDN layers.
        for (int gidx = 0; gidx < 4; ++gidx) {
            double lss = 0.0;
            for (int i = 0; i < 48 * 128 * 128; ++i) {
                double v = gdn_state[(std::size_t)gidx * 48 * 128 * 128 + i];
                lss += v * v;
            }
            std::fprintf(stderr, "[diag]   GDN layer gidx=%d state rms=%.5f\n", gidx, std::sqrt(lss / (48 * 128 * 128)));
        }
    }

    // ---- RAW SCALE AUDIT: confirm inflated scales are real in the file ----
    auto raw_scale_dump = [&](const std::string& name, int row, int ngroups) {
        const auto& t = store.get(name);
        const std::uint64_t shape[2] = {t.shape[0], t.shape[1]};
        auto geo = row_split_geometry(t.format, std::span<const std::uint64_t>(shape, 2));
        const int gpr = (int)geo.groups_per_row;
        const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(t.bytes.data() + geo.scale_plane_offset);
        double smn = 1e30, smx = -1e30, sss = 0; int sn = 0;
        for (int g = 0; g < gpr; ++g) {
            float s = f16_to_f32(scale[(std::size_t)row * gpr + g]);
            smn = std::min(smn, (double)s); smx = std::max(smx, (double)s); sss += (double)s; sn++;
        }
        std::fprintf(stderr, "[audit] %s row %d: scale min=%.6f max=%.6f mean=%.6f | raw[0:%d]:", name.c_str(), row, smn, smx, sss / sn, ngroups);
        for (int g = 0; g < ngroups && g < gpr; ++g) {
            std::uint16_t sb = scale[(std::size_t)row * gpr + g];
            std::fprintf(stderr, " %04x=%.5f", sb, f16_to_f32(sb));
        }
        std::fprintf(stderr, "\n");
    };
    raw_scale_dump("text/token_embedding", 11, 12);
    raw_scale_dump("text/token_embedding", 0, 12);
    raw_scale_dump("text/output_head", 133946, 12);
    raw_scale_dump("text/output_head", 0, 12);
    raw_scale_dump("text/layers/0/gdn/output", 3994, 12);
    raw_scale_dump("text/layers/0/gdn/output", 0, 12);
    raw_scale_dump("text/layers/0/mlp/down", 3994, 12);
    raw_scale_dump("text/layers/0/mlp/down", 0, 12);

    // logits
    std::vector<std::uint16_t> h;
    rmsnorm(x, store.dense_bf16("text/final_norm"), h, C::hidden);
    // Diagnostic: rms/max of x, final_norm weight, and h.
    auto stats16 = [](const std::vector<std::uint16_t>& v) {
        double ss = 0, mn = 1e30, mx = -1e30;
        for (auto b : v) { double f = from_bf16(b); ss += f * f; mn = std::min(mn, f); mx = std::max(mx, f); }
        return std::make_tuple(std::sqrt(ss / v.size()), mn, mx);
    };
    auto [xrms, xmn, xmx] = stats16(x);
    auto [hrms, hmn, hmx] = stats16(h);
    const auto* fnw = store.dense_bf16("text/final_norm");
    double fnss = 0; for (int i = 0; i < C::hidden; ++i) { double f = from_bf16(fnw[i]); fnss += f * f; }
    std::fprintf(stderr, "[diag] x rms=%.4f min=%.3f max=%.3f | final_norm rms=%.4f | h rms=%.4f min=%.3f max=%.3f\n",
                 xrms, xmn, xmx, std::sqrt(fnss / C::hidden), hrms, hmn, hmx);
    std::vector<double> logits;
    store.gemv("text/output_head", to_double(h), logits);
    std::vector<std::pair<double, int>> top;
    for (int i = 0; i < C::token_domain; ++i) top.emplace_back(logits[i], i);
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "[cpu] logits top5:");
    for (int i = 0; i < 5; ++i) std::fprintf(stderr, " (%d, %.3f)", top[i].second, top[i].first);
    std::fprintf(stderr, "\n");
    // Masked argmax over valid tokens only.
    int best = 0; double best_v = logits[0];
    for (int i = 1; i < C::token_domain; ++i)
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    std::fprintf(stderr, "[cpu] argmax (valid) = %d (%.3f)\n", best, best_v);
    // Where does the expected "We" (1596) rank?
    std::fprintf(stderr, "[cpu] logit[1596 'We'] = %.3f\n", logits[1596]);
    // Top-10 for context.
    std::vector<std::pair<double, int>> top10;
    for (int i = 0; i < C::token_domain; ++i) top10.emplace_back(logits[i], i);
    std::partial_sort(top10.begin(), top10.begin() + 10, top10.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "[cpu] top10:");
    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " (%d, %.2f)", top10[i].second, top10[i].first);
    std::fprintf(stderr, "\n");
    return 0;
}