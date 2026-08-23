// Verify the .ninfer v2 reader against the real artifact.
//
// Opens the artifact, prints identity + object counts, and re-derives every
// tensor's encoded size from its (layout, format, shape) to confirm the stored
// byte count matches. Any mismatch means the geometry port is wrong.

#include "artifact/reader.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace nsycl::artifact;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <artifact.ninfer>\n", argv[0]);
        return 2;
    }

    Reader reader(argv[1]);
    const auto& id = reader.identity();
    std::printf("identity: model_id=%s weights_id=%s\n", id.model_id.c_str(),
                id.weights_id.c_str());
    std::printf("file_bytes=%llu payload_offset=%llu objects=%zu\n",
                (unsigned long long)reader.file_bytes(),
                (unsigned long long)reader.payload_offset(), reader.objects().size());

    std::unordered_map<std::string, int> by_format;
    std::uint64_t total_bytes = 0;
    int tensors = 0, resources = 0, mismatches = 0;

    for (const auto& obj : reader.objects()) {
        const auto name = std::string(object_name(obj));
        if (const auto* t = std::get_if<TensorDescriptor>(&obj)) {
            ++tensors;
            total_bytes += t->bytes;
            by_format[std::string(format_name(t->format))]++;
            const auto expected =
                tensor_encoded_size(t->layout, t->format,
                                    std::span<const std::uint64_t>(t->shape.data(), t->shape.size()));
            if (expected != t->bytes) {
                ++mismatches;
                std::fprintf(stderr, "MISMATCH %s: stored=%llu computed=%llu\n", name.c_str(),
                             (unsigned long long)t->bytes, (unsigned long long)expected);
            }
        } else {
            ++resources;
            total_bytes += object_bytes(obj);
        }
    }

    std::printf("tensors=%d resources=%d total_payload_bytes=%llu (%.3f GiB)\n", tensors,
                resources, (unsigned long long)total_bytes,
                (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
    std::printf("by format:\n");
    for (const auto& [fmt, n] : by_format) {
        std::printf("  %-22s %d\n", fmt.c_str(), n);
    }

    if (mismatches != 0) {
        std::fprintf(stderr, "FAILED: %d geometry mismatches\n", mismatches);
        return 1;
    }
    std::printf("OK: all tensor geometries match\n");
    return 0;
}