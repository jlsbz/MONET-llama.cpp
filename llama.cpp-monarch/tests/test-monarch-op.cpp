#include "llama-monarch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static std::vector<float> reference_monarch(
        const std::vector<float> & input,
        const std::vector<float> & left,
        const std::vector<float> & right,
        const std::vector<int32_t> & perm,
        int64_t width,
        int64_t n_rows,
        int64_t block_size) {
    const int64_t num_blocks = width / block_size;
    std::vector<float> result(input.size());
    std::vector<float> after_right(width);

    for (int64_t row = 0; row < n_rows; ++row) {
        const float * x = input.data() + row * width;
        float * y = result.data() + row * width;

        for (int64_t block = 0; block < num_blocks; ++block) {
            for (int64_t out = 0; out < block_size; ++out) {
                float sum = 0.0f;
                for (int64_t in = 0; in < block_size; ++in) {
                    const int64_t index = block * block_size * block_size + in * block_size + out;
                    sum += x[block * block_size + in] * right[index];
                }
                after_right[block * block_size + out] = sum;
            }
        }

        for (int64_t block = 0; block < num_blocks; ++block) {
            for (int64_t out = 0; out < block_size; ++out) {
                float sum = 0.0f;
                for (int64_t in = 0; in < block_size; ++in) {
                    const int64_t index = block * block_size * block_size + in * block_size + out;
                    sum += after_right[perm[block * block_size + in]] * left[index];
                }
                y[block * block_size + out] = sum;
            }
        }
    }

    return result;
}

static void run_case(ggml_type factor_type) {
    constexpr int64_t block_size = 2;
    constexpr int64_t num_blocks = 4;
    constexpr int64_t width = block_size * num_blocks;
    constexpr int64_t n_rows = 3;

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, n_rows);
    ggml_tensor * left  = ggml_new_tensor_3d(ctx, factor_type, block_size, block_size, num_blocks);
    ggml_tensor * right = ggml_new_tensor_3d(ctx, factor_type, block_size, block_size, num_blocks);
    ggml_tensor * perm  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, width);
    ggml_tensor * output = llama_monarch_linear(ctx, input, left, right, perm);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> input_data(width * n_rows);
    std::vector<float> left_data(block_size * block_size * num_blocks);
    std::vector<float> right_data(left_data.size());
    const std::vector<int32_t> perm_data = { 2, 0, 7, 4, 1, 6, 3, 5 };

    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = 0.05f * static_cast<float>(static_cast<int>(i) - 7);
    }
    for (size_t i = 0; i < left_data.size(); ++i) {
        left_data[i] = 0.03f * static_cast<float>(static_cast<int>(i % 9) - 4);
        right_data[i] = 0.04f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(perm, perm_data.data(), 0, perm_data.size() * sizeof(int32_t));

    if (factor_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(left, left_data.data(), 0, left_data.size() * sizeof(float));
        ggml_backend_tensor_set(right, right_data.data(), 0, right_data.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> left_f16(left_data.size());
        std::vector<ggml_fp16_t> right_f16(right_data.size());
        for (size_t i = 0; i < left_data.size(); ++i) {
            left_f16[i] = ggml_fp32_to_fp16(left_data[i]);
            right_f16[i] = ggml_fp32_to_fp16(right_data[i]);
            left_data[i] = ggml_fp16_to_fp32(left_f16[i]);
            right_data[i] = ggml_fp16_to_fp32(right_f16[i]);
        }
        ggml_backend_tensor_set(left, left_f16.data(), 0, left_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(right, right_f16.data(), 0, right_f16.size() * sizeof(ggml_fp16_t));
    }

    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(input_data.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    const auto expected = reference_monarch(
        input_data, left_data, right_data, perm_data, width, n_rows, block_size);

    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    assert(max_error < 1e-6f);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    run_case(GGML_TYPE_F32);
    run_case(GGML_TYPE_F16);
    std::puts("MONET Monarch custom OP tests passed");
    return 0;
}
