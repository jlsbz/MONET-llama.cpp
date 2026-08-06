#include "llama.h"
#include "ggml.h"

#include <cstdlib>
#include <cstdio>

static bool observe_graph(ggml_tensor * tensor, bool, void * user_data) {
    if (tensor->op == GGML_OP_CUSTOM) {
        ++*static_cast<int *>(user_data);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s model.gguf [minimum-custom-op-count]\n", argv[0]);
        return 2;
    }

    const int minimum_custom_ops = argc == 3 ? std::atoi(argv[2]) : 1;
    if (minimum_custom_ops < 0) {
        std::fprintf(stderr, "minimum custom OP count must be non-negative\n");
        return 2;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load MONET model\n");
        llama_backend_free();
        return 1;
    }

    int custom_op_count = 0;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.cb_eval = observe_graph;
    context_params.cb_eval_user_data = &custom_op_count;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create MONET context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_token token = 1;
    llama_batch batch = llama_batch_get_one(&token, 1);
    const int decode_result = llama_decode(context, batch);

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();

    if (decode_result != 0) {
        std::fprintf(stderr, "MONET decode failed: %d\n", decode_result);
        return 1;
    }
    if (custom_op_count < minimum_custom_ops) {
        std::fprintf(
            stderr,
            "decode succeeded but observed only %d GGML_OP_CUSTOM node(s), expected at least %d\n",
            custom_op_count,
            minimum_custom_ops);
        return 1;
    }

    std::printf(
        "MONET model load/decode/custom-OP test passed (observed %d custom OP node(s))\n",
        custom_op_count);
    return 0;
}
