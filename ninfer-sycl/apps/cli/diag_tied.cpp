// Check if token_embedding and output_head are tied (identical bytes).
#include "artifact/reader.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

using namespace nsycl::artifact;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact>\n", argv[0]); return 1; }
    Reader store{std::filesystem::path(argv[1])};
    const auto* emb = store.find("text/token_embedding");
    const auto* head = store.find("text/output_head");
    if (!emb || !head) { std::fprintf(stderr, "missing tensor\n"); return 1; }
    auto se = store.payload(*emb);
    auto sh = store.payload(*head);
    std::fprintf(stderr, "emb bytes=%zu head bytes=%zu\n", se.data.size(), sh.data.size());
    if (se.data.size() != sh.data.size()) { std::fprintf(stderr, "size mismatch\n"); return 1; }
    const auto* pe = reinterpret_cast<const std::uint8_t*>(se.data.data());
    const auto* ph = reinterpret_cast<const std::uint8_t*>(sh.data.data());
    std::size_t diff = 0, first_diff = 0;
    for (std::size_t i = 0; i < se.data.size(); ++i) {
        if (pe[i] != ph[i]) { if (diff == 0) first_diff = i; ++diff; }
    }
    std::fprintf(stderr, "differing bytes: %zu / %zu (first at %zu)\n", diff, se.data.size(), first_diff);
    if (diff == 0) std::fprintf(stderr, "=> TIED (identical)\n");
    else std::fprintf(stderr, "=> NOT tied\n");
    return 0;
}