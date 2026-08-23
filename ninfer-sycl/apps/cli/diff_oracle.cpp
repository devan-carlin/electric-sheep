// diff_oracle.cpp - feed the exact 53-token "Hello" prompt to llama.cpp (SYCL),
// greedy-decode one token, print top-5 next tokens + logits.
// Expected first token: 1596 "We".
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static llama_token PROMPT[] = {
    248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13,
    5044, 1683, 15060, 1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814,
    47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321, 30246, 303,
    279, 1534, 4087, 13, 248046, 198, 248045, 846, 198, 9419, 248046, 198,
    248045, 74455, 198, 248068, 198
};
static const int NPROMPT = (int)(sizeof(PROMPT) / sizeof(PROMPT[0]));

int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const char* model_path = argv[1];

    llama_backend_init();
    llama_print_system_info();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1; // all layers on GPU
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE; // single GPU, no tensor parallelism
    mparams.main_gpu = 0;
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "FATAL: model load failed\n"); return 2; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "FATAL: ctx init failed\n"); return 3; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::fprintf(stderr, "[oracle] n_vocab=%d nprompt=%d\n", n_vocab, NPROMPT);

    // Prefill the 53-token prompt.
    std::fprintf(stderr, "[oracle] step: batch_get_one\n");
    llama_batch batch = llama_batch_get_one(PROMPT, NPROMPT);
    std::fprintf(stderr, "[oracle] step: decode (prefill)\n");
    int rc = llama_decode(ctx, batch);
    std::fprintf(stderr, "[oracle] prefill rc=%d\n", rc);
    std::fprintf(stderr, "[oracle] step: batch_free\n");
    llama_batch_free(batch);
    std::fprintf(stderr, "[oracle] step: batch_free done\n");
    if (rc < 0) { std::fprintf(stderr, "FATAL: prefill failed rc=%d\n", rc); return 4; }

    // Logits for the last prompt position. llama_batch_get_one only requests
    // logits for the final position, so llama_get_logits() returns exactly that.
    std::fprintf(stderr, "[oracle] step: get_logits\n");
    std::fflush(stderr);
    float* logits = llama_get_logits(ctx);
    if (!logits) { std::fprintf(stderr, "FATAL: no logits\n"); return 5; }
    std::fprintf(stderr, "[oracle] step: logits ok, building idx\n");
    std::fflush(stderr);

    // Top-5 by logit.
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    std::fprintf(stderr, "[oracle] step: sort ok, decoding tokens\n");
    std::fflush(stderr);

    std::fprintf(stderr, "[oracle] top-5 next tokens:\n");
    for (int r = 0; r < 5; ++r) {
        int t = idx[r];
        char buf[256];
        int n = llama_token_to_piece(llama_model_get_vocab(model), t, buf, sizeof(buf), 0, false);
        if (n < 0) std::snprintf(buf, sizeof(buf), "<err>");
        std::fprintf(stderr, "  #%d  tok=%d  logit=%.4f  text=%s\n", r + 1, t, logits[t], buf);
    }

    int best = idx[0];
    std::fprintf(stderr, "[oracle] ARGMAX=%d  (expected 1596 \"We\")  %s\n",
                 best, best == 1596 ? "MATCH" : "MISMATCH");
    std::fflush(stderr);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}