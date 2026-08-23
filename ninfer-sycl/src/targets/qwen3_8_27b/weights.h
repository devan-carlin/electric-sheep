#pragma once

// nsycl::model - weight store for the Qwen3.8-27B text model.
//
// Loads every text tensor from the .ninfer artifact into shared USM (device-
// readable). Quantized tensors are copied as a single buffer; the 3 row-split
// planes (low codes, high bits, f16 scales) are offsets into it. Provides a
// GEMV dispatcher and dense-tensor accessors for the forward pass.

#include "artifact/reader.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nsycl::model {

class Weights {
public:
    Weights();
    explicit Weights(sycl::queue& q, const std::filesystem::path& artifact_path);
    ~Weights();

    Weights(const Weights&)            = delete;
    Weights& operator=(const Weights&) = delete;
    Weights(Weights&&) noexcept;
    Weights& operator=(Weights&&) noexcept;

    // GEMV: out[n] = sum_k W[n,k] * x[k]. x is bf16 [K], out is f32 [N].
    // name is the artifact tensor name (e.g. "text/layers/0/mlp/gate_up").
    void gemv(const std::string& name, const std::uint16_t* x, float* out,
              sycl::queue& q) const;

    // Dense BF16 GEMV for contiguous-le-v1 BF16 tensors (e.g. gdn/a_projection).
    void gemv_bf16(const std::string& name, const std::uint16_t* x, float* out,
                   sycl::queue& q) const;

    // W8 row dequant: dequantize one row of a W8G32 tensor to bf16 (embedding).
    void w8_row_dequant(const std::string& name, int row, std::uint16_t* out,
                        sycl::queue& q) const;

    // Dense tensor accessors (return device pointers).
    const std::uint16_t* bf16(const std::string& name) const;
    const float* fp32(const std::string& name) const;
    const std::int32_t* i32(const std::string& name) const;

    // Element count of a dense tensor.
    std::size_t dense_elements(const std::string& name) const;

    // Shape of a quantized tensor (rows, cols).
    void quant_shape(const std::string& name, int& rows, int& cols) const;

private:
    struct QuantEntry {
        const std::uint8_t* low   = nullptr;
        const std::uint8_t* high  = nullptr;
        const std::uint16_t* scale = nullptr;
        int N = 0, K = 0, groups_per_row = 0;
        artifact::NumericFormat format = artifact::NumericFormat::BF16;
    };

    struct DenseEntry {
        const void* ptr = nullptr;
        std::size_t bytes = 0;
        artifact::NumericFormat format = artifact::NumericFormat::BF16;
        int N = 0; // rows (for 2-D dense)
        int K = 0; // cols (for 2-D dense)
    };

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nsycl::model