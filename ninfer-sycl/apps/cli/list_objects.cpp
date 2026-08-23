// List all objects (tensors + resources) in a .ninfer artifact, and optionally
// dump a named resource to a file.
//
// Usage:
//   list_objects <artifact.ninfer>
//   list_objects <artifact.ninfer> --dump <resource_name> <out_path>

#include "artifact/reader.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

using namespace nsycl::artifact;

int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact.ninfer> [--dump <name> <out>]\n", argv[0]); return 1; }
    std::fprintf(stderr, "opening %s\n", argv[1]);
    Reader reader(argv[1]);
    std::fprintf(stderr, "reader ok, %zu objects\n", reader.objects().size());

    std::string dump_name, dump_out;
    if (argc >= 5 && std::strcmp(argv[2], "--dump") == 0) {
        dump_name = argv[3];
        dump_out  = argv[4];
    }

    std::size_t nres = 0, ntens = 0;
    for (const auto& obj : reader.objects()) {
        const auto name = std::string(object_name(obj));
        if (const auto* t = std::get_if<TensorDescriptor>(&obj)) {
            ++ntens;
            if (name.rfind("text/", 0) == 0) {
                std::fprintf(stderr, "TENSOR %-50s fmt=%s layout=%s shape=[", name.c_str(),
                             format_name(t->format).data(), layout_name(t->layout).data());
                for (size_t i = 0; i < t->shape.size(); ++i)
                    std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)t->shape[i]);
                std::fprintf(stderr, "] bytes=%llu\n", (unsigned long long)t->bytes);
            }
        } else if (const auto* r = std::get_if<ResourceDescriptor>(&obj)) {
            ++nres;
            std::fprintf(stderr, "RESOURCE %-50s enc=%s bytes=%llu\n", name.c_str(),
                         encoding_name(r->encoding).data(), (unsigned long long)r->bytes);
            if (!dump_name.empty() && name == dump_name) {
                auto span = reader.payload(obj);
                std::ofstream f(dump_out, std::ios::binary);
                f.write(reinterpret_cast<const char*>(span.data.data()), span.data.size());
                std::fprintf(stderr, "  -> dumped %zu bytes to %s\n", span.data.size(), dump_out.c_str());
            }
        }
    }
    std::fprintf(stderr, "total: %zu tensors, %zu resources\n", ntens, nres);
    std::fflush(stderr);
    return 0;
}