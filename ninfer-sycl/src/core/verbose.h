#pragma once

// nsycl::core - toggleable verbose logging for debugging.
//
// Enable by setting the environment variable NINFER_VERBOSE to any value other
// than "0" or empty (e.g. NINFER_VERBOSE=1). The variable is read once and
// cached, so toggling it at runtime has no effect; set it before launch.
//
// All output goes to stderr with a "[verbose]" prefix.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace nsycl {

// Verbose level: 0 off, 1 basic, 2 + per-GEMV trace, 3 + per-layer/persistent
// state trace, 4 + weights USM kernel readback.
[[nodiscard]] inline int verbose_level() noexcept {
    static const int level = [] {
        const char* v = std::getenv("NINFER_VERBOSE");
        if (v == nullptr || v[0] == '\0') return 0;
        int l = std::atoi(v);
        return l < 0 ? 0 : (l > 4 ? 4 : l);
    }();
    return level;
}

[[nodiscard]] inline bool verbose_enabled() noexcept { return verbose_level() >= 1; }

// Parse a comma-separated int list from an env var (empty if unset).
[[nodiscard]] inline std::vector<int> parse_int_list(const char* env) {
    std::vector<int> out;
    const char* v = std::getenv(env);
    if (v == nullptr || v[0] == '\0') return out;
    std::string s(v);
    std::size_t start = 0;
    while (start < s.size()) {
        std::size_t comma = s.find(',', start);
        std::string tok = (comma == std::string::npos) ? s.substr(start)
                                                       : s.substr(start, comma - start);
        std::size_t a = tok.find_first_not_of(" \t");
        std::size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos && b != std::string::npos) {
            out.push_back(std::atoi(tok.substr(a, b - a + 1).c_str()));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

// True if `layer` is in NINFER_LOG_LAYERS (all if the filter is unset).
[[nodiscard]] inline bool log_layer(int layer) noexcept {
    static const std::vector<int> layers = parse_int_list("NINFER_LOG_LAYERS");
    if (layers.empty()) return true;
    for (int l : layers) if (l == layer) return true;
    return false;
}

// True if `token` is in NINFER_LOG_TOKENS (all if the filter is unset).
[[nodiscard]] inline bool log_token(int token) noexcept {
    static const std::vector<int> tokens = parse_int_list("NINFER_LOG_TOKENS");
    if (tokens.empty()) return true;
    for (int t : tokens) if (t == token) return true;
    return false;
}

// Extract the layer index from a tensor name like "text/layers/7/gdn/query_key".
// Returns -1 if the name has no "layers/N/" segment.
[[nodiscard]] inline int layer_from_name(const std::string& name) noexcept {
    const std::size_t p = name.find("layers/");
    if (p == std::string::npos) return -1;
    std::size_t i = p + 7;
    std::size_t j = i;
    while (j < name.size() && name[j] >= '0' && name[j] <= '9') ++j;
    if (j == i) return -1;
    return std::atoi(name.substr(i, j - i).c_str());
}

// True if the tensor name's layer passes the NINFER_LOG_LAYERS filter.
[[nodiscard]] inline bool log_layer_name(const std::string& name) noexcept {
    int l = layer_from_name(name);
    if (l < 0) return true; // non-layer tensor (embedding, output_head): always log
    return log_layer(l);
}

} // namespace nsycl

#define NINFER_VERBOSE(...) \
    do { \
        if (::nsycl::verbose_enabled()) { \
            std::fprintf(stderr, "[verbose] " __VA_ARGS__); \
            std::fputc('\n', stderr); \
        } \
    } while (0)

#define NINFER_VERBOSE_L(lvl, ...) \
    do { \
        if (::nsycl::verbose_level() >= (lvl)) { \
            std::fprintf(stderr, "[verbose] " __VA_ARGS__); \
            std::fputc('\n', stderr); \
        } \
    } while (0)