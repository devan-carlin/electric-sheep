#include "targets/qwen3_8_27b/weights.h"

#include "core/verbose.h"
#include "ops/linear/gemv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nsycl::model {

namespace {
// Host rms of a bf16 buffer (device -> host copy).
double rms_bf16(sycl::queue& q, const std::uint16_t* p, int n) {
    std::vector<std::uint16_t> h(n);
    q.memcpy(h.data(), p, n * 2);
    q.wait();
    double ss = 0.0;
    for (auto v : h) { float f = ops::bf16_to_f32(v); ss += (double)f * f; }
    return std::sqrt(ss / n);
}
// Host rms + first 4 of an f32 buffer (device -> host copy).
void rms_f32(sycl::queue& q, const float* p, int n, double& rms, float* first4) {
    std::vector<float> h(n);
    q.memcpy(h.data(), p, n * 4);
    q.wait();
    double ss = 0.0;
    for (float v : h) ss += (double)v * v;
    rms = std::sqrt(ss / n);
    for (int i = 0; i < 4 && i < n; ++i) first4[i] = h[i];
}
} // namespace

struct Weights::Impl {
    std::vector<std::uint8_t*> usm_buffers;
    std::unordered_map<std::string, QuantEntry> quant;
    std::unordered_map<std::string, DenseEntry> dense;
    std::size_t total_bytes = 0;
};

Weights::Weights() = default;

Weights::Weights(sycl::queue& q, const std::filesystem::path& artifact_path)
    : impl_(std::make_unique<Impl>()) {
    artifact::Reader reader(artifact_path);
    NINFER_VERBOSE("Weights: loading %s", artifact_path.string().c_str());

    // Pass 1: find the largest (offset + bytes) among text/mtp tensors to size
    // one big USM buffer.
    std::uint64_t max_end = 0;
    for (const auto& obj : reader.objects()) {
        const auto name = std::string(artifact::object_name(obj));
        if (name.rfind("text/", 0) != 0 && name.rfind("mtp/", 0) != 0) { continue; }
        const auto end = artifact::object_offset(obj) + artifact::object_bytes(obj);
        if (end > max_end) { max_end = end; }
    }

    std::uint8_t* base = sycl::malloc_device<std::uint8_t>(max_end, q);
    impl_->usm_buffers.push_back(base);
    impl_->total_bytes = max_end;

    // Pass 2: copy each text/mtp tensor into the buffer at its offset.
    for (const auto& obj : reader.objects()) {
        const auto name = std::string(artifact::object_name(obj));
        if (name.rfind("text/", 0) != 0 && name.rfind("mtp/", 0) != 0) { continue; }

        const auto offset = artifact::object_offset(obj);
        const auto bytes  = artifact::object_bytes(obj);
        const auto* src   = reader.payload(obj).data.data();
        q.memcpy(base + offset, src, bytes);

        if (const auto* t = std::get_if<artifact::TensorDescriptor>(&obj)) {
            if (t->layout == artifact::StorageLayout::RowSplitK128V1) {
                const std::uint64_t shape[2] = {t->shape[0], t->shape[1]};
                auto geo = artifact::row_split_geometry(
                    t->format, std::span<const std::uint64_t>(shape, 2));
                QuantEntry e;
                e.low   = base + offset;
                e.high  = base + offset + geo.high_plane_offset;
                e.scale = reinterpret_cast<const std::uint16_t*>(base + offset + geo.scale_plane_offset);
                e.N     = static_cast<int>(t->shape[0]);
                e.K     = static_cast<int>(t->shape[1]);
                e.groups_per_row = static_cast<int>(geo.groups_per_row);
                e.format = t->format;
                impl_->quant.emplace(name, e);
            } else {
                DenseEntry e;
                e.ptr    = base + offset;
                e.bytes  = bytes;
                e.format = t->format;
                if (t->shape.size() == 2) {
                    e.N = static_cast<int>(t->shape[0]);
                    e.K = static_cast<int>(t->shape[1]);
                }
                impl_->dense.emplace(name, e);
            }
        }
    }
    q.wait();
    NINFER_VERBOSE("Weights: %zu quant + %zu dense tensors, %.3f GiB USM",
                   impl_->quant.size(), impl_->dense.size(),
                   (double)impl_->total_bytes / (1024.0 * 1024.0 * 1024.0));

    // Level-4 diagnostic: read the scale planes of a few quant tensors back
    // through a KERNEL (not memcpy) to test whether the weights USM is coherent
    // for kernel reads. If a kernel sees zero/stale scales here, the USM read
    // path is the problem, not the GEMV math.
    if (nsycl::verbose_level() >= 4) {
        for (const char* probe : {"text/layers/0/gdn/query_key",
                                  "text/layers/0/gdn/value_z",
                                  "text/layers/3/attention/query_key"}) {
            const auto it = impl_->quant.find(probe);
            if (it == impl_->quant.end()) continue;
            const QuantEntry& e = it->second;
            const int ngroups = e.N * e.groups_per_row;
            std::vector<float> host(ngroups);
            float* host_ptr = host.data();
            const std::uint16_t* scale_ptr = e.scale;
            q.parallel_for(sycl::range<1>(ngroups), [=](sycl::id<1> id) {
                host_ptr[id[0]] = ops::f16_to_f32(scale_ptr[id[0]]);
            });
            q.wait();
            double ss = 0.0;
            int zeros = 0;
            for (float v : host) { ss += (double)v * v; if (v == 0.0f) ++zeros; }
            NINFER_VERBOSE("Weights USM readback %s: scale rms=%.6f zeros=%d/%d",
                           probe, std::sqrt(ss / ngroups), zeros, ngroups);
        }
    }
}

Weights::~Weights() {
    // USM buffers are freed by the queue on destruction; nothing to do here.
}

Weights::Weights(Weights&&) noexcept = default;
Weights& Weights::operator=(Weights&&) noexcept = default;

void Weights::gemv(const std::string& name, const std::uint16_t* x, float* out,
                   sycl::queue& q) const {
    const auto it = impl_->quant.find(name);
    if (it == impl_->quant.end()) {
        throw std::runtime_error("gemv: unknown quant tensor " + name);
    }
    const QuantEntry& e = it->second;
    switch (e.format) {
    case artifact::NumericFormat::Q4G64_F16S:
        ops::gemv_q4(q, e.low, e.scale, x, out, e.N, e.K, e.groups_per_row);
        break;
    case artifact::NumericFormat::Q5G64_F16S:
        ops::gemv_q5(q, e.low, e.high, e.scale, x, out, e.N, e.K, e.groups_per_row);
        break;
    case artifact::NumericFormat::Q6G64_F16S:
        ops::gemv_q6(q, e.low, e.high, e.scale, x, out, e.N, e.K, e.groups_per_row);
        break;
    case artifact::NumericFormat::W8G32_F16S:
        ops::gemv_w8(q, e.low, e.scale, x, out, e.N, e.K, e.groups_per_row);
        break;
    default:
        throw std::runtime_error("gemv: unsupported format for " + name);
    }
    q.wait(); // USM coherence: force GEMV output visible before the next kernel reads it.
    if (nsycl::verbose_level() >= 2 && nsycl::log_layer_name(name)) {
        double xrms = 0.0, orms = 0.0;
        float of4[4] = {0, 0, 0, 0};
        xrms = rms_bf16(q, x, e.K);
        rms_f32(q, out, e.N, orms, of4);
        NINFER_VERBOSE("GEMV %-40s in rms=%.5f out rms=%.5f out[0:4]=%.4f,%.4f,%.4f,%.4f",
                       name.c_str(), xrms, orms, of4[0], of4[1], of4[2], of4[3]);
    }
}

void Weights::gemv_bf16(const std::string& name, const std::uint16_t* x, float* out,
                        sycl::queue& q) const {
    const auto it = impl_->dense.find(name);
    if (it == impl_->dense.end() || it->second.format != artifact::NumericFormat::BF16) {
        throw std::runtime_error("gemv_bf16: unknown or wrong-format tensor " + name);
    }
    const DenseEntry& e = it->second;
    ops::gemv_bf16(q, static_cast<const std::uint16_t*>(e.ptr), x, out, e.N, e.K);
    q.wait(); // USM coherence: force GEMV output visible before the next kernel reads it.
    if (nsycl::verbose_level() >= 2 && nsycl::log_layer_name(name)) {
        double xrms = 0.0, orms = 0.0;
        float of4[4] = {0, 0, 0, 0};
        xrms = rms_bf16(q, x, e.K);
        rms_f32(q, out, e.N, orms, of4);
        NINFER_VERBOSE("GEMV %-40s in rms=%.5f out rms=%.5f out[0:4]=%.4f,%.4f,%.4f,%.4f",
                       name.c_str(), xrms, orms, of4[0], of4[1], of4[2], of4[3]);
    }
}

void Weights::w8_row_dequant(const std::string& name, int row, std::uint16_t* out,
                             sycl::queue& q) const {
    const auto it = impl_->quant.find(name);
    if (it == impl_->quant.end() || it->second.format != artifact::NumericFormat::W8G32_F16S) {
        throw std::runtime_error("w8_row_dequant: unknown or wrong-format tensor " + name);
    }
    const QuantEntry& e = it->second;
    ops::w8_row_dequant(q, e.low, e.scale, out, row, e.K, e.groups_per_row);
}

const std::uint16_t* Weights::bf16(const std::string& name) const {
    const auto it = impl_->dense.find(name);
    if (it == impl_->dense.end() || it->second.format != artifact::NumericFormat::BF16) {
        throw std::runtime_error("bf16: unknown or wrong-format tensor " + name);
    }
    return static_cast<const std::uint16_t*>(it->second.ptr);
}

const float* Weights::fp32(const std::string& name) const {
    const auto it = impl_->dense.find(name);
    if (it == impl_->dense.end() || it->second.format != artifact::NumericFormat::FP32) {
        throw std::runtime_error("fp32: unknown or wrong-format tensor " + name);
    }
    return static_cast<const float*>(it->second.ptr);
}

const std::int32_t* Weights::i32(const std::string& name) const {
    const auto it = impl_->dense.find(name);
    if (it == impl_->dense.end() || it->second.format != artifact::NumericFormat::I32) {
        throw std::runtime_error("i32: unknown or wrong-format tensor " + name);
    }
    return static_cast<const std::int32_t*>(it->second.ptr);
}

std::size_t Weights::dense_elements(const std::string& name) const {
    const auto it = impl_->dense.find(name);
    if (it == impl_->dense.end()) { throw std::runtime_error("dense_elements: unknown " + name); }
    const int word = (it->second.format == artifact::NumericFormat::BF16) ? 2 : 4;
    return it->second.bytes / static_cast<std::size_t>(word);
}

void Weights::quant_shape(const std::string& name, int& rows, int& cols) const {
    const auto it = impl_->quant.find(name);
    if (it == impl_->quant.end()) { throw std::runtime_error("quant_shape: unknown " + name); }
    rows = it->second.N;
    cols = it->second.K;
}

} // namespace nsycl::model