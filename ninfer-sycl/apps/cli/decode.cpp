// nsycl decode app: load the model, prefill a prompt, greedy-decode N tokens.
//
// Usage: decode <artifact.ninfer> <max_new_tokens> [tok tok ...]
// If no token ids are given, a default prompt is used.

#include "targets/qwen3_8_27b/model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <artifact.ninfer> <max_new_tokens> [tok ...]\n", argv[0]);
        return 1;
    }
    std::filesystem::path artifact = argv[1];
    int max_new = std::atoi(argv[2]);

    std::vector<int> prompt;
    for (int i = 3; i < argc; ++i) {
        prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty()) {
        // Default prompt: "Hello" (token id 9707 in Qwen3 tokenizer).
        prompt = {9707};
    }

    sycl::queue q{sycl::default_selector{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<int> prompt_host = prompt;
    std::printf("prompt (%zu tokens):", prompt_host.size());
    for (int t : prompt_host) std::printf(" %d", t);
    std::printf("\n");

    nsycl::model::Model model(q, artifact);

    // Warm up the queue / device. The first few kernel launches on a fresh
    // SYCL queue can misbehave on the Level Zero backend; a short burst of
    // dummy work avoids it.
    {
        auto* warm = sycl::malloc_device<float>(1024, q);
        for (int i = 0; i < 16; ++i) {
            q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> id) {
                warm[id[0]] = static_cast<float>(id[0] + i);
            });
        }
        q.wait();
        sycl::free(warm, q);
    }

    // Prefill (returns the first generated token).
    int first = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        first = model.prefill(prompt, q);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("prefill: %zu tokens in %.1f ms, first=%d\n", prompt.size(), ms, first);
    }

    // Decode.
    std::vector<int> out;
    out.push_back(first);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 1; i < max_new; ++i) {
        int tok = model.decode(q);
        out.push_back(tok);
        std::printf("%d\n", tok);
        std::fflush(stdout);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("decode: %zu tokens in %.1f ms (%.1f tok/s)\n", out.size(), ms,
                1000.0 * out.size() / ms);
    return 0;
}