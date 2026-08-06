#include "llama-monarch.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

static float llama_monarch_factor_at(const ggml_tensor * tensor, int64_t index) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return static_cast<const float *>(tensor->data)[index];
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(tensor->data)[index]);
        default:
            GGML_ABORT("unsupported Monarch factor type: %s", ggml_type_name(tensor->type));
    }
}

static void llama_monarch_compute(
        ggml_tensor * dst,
                  int ith,
                  int nth,
               void *) {
    const ggml_tensor * input       = dst->src[0];
    const ggml_tensor * left        = dst->src[1];
    const ggml_tensor * right       = dst->src[2];
    const ggml_tensor * permutation = dst->src[3];

    const int64_t block_size = left->ne[0];
    const int64_t num_blocks = left->ne[2];
    const int64_t width      = block_size * num_blocks;
    const int64_t n_rows     = ggml_nrows(input);
    const auto * perm        = static_cast<const int32_t *>(permutation->data);

    // One scratch vector per worker. This keeps the portable reference kernel
    // independent of GGML's graph workspace allocation.
    thread_local std::vector<float> after_right;
    after_right.resize(width);

    for (int64_t row = ith; row < n_rows; row += nth) {
        const auto * x = reinterpret_cast<const float *>(
            static_cast<const char *>(input->data) + row * input->nb[1]);
        auto * y = reinterpret_cast<float *>(
            static_cast<char *>(dst->data) + row * dst->nb[1]);

        for (int64_t block = 0; block < num_blocks; ++block) {
            const int64_t matrix_offset = block * block_size * block_size;
            const int64_t vector_offset = block * block_size;

            for (int64_t out = 0; out < block_size; ++out) {
                float sum = 0.0f;
                for (int64_t in = 0; in < block_size; ++in) {
                    sum += x[vector_offset + in] *
                           llama_monarch_factor_at(right, matrix_offset + in * block_size + out);
                }
                after_right[vector_offset + out] = sum;
            }
        }

        for (int64_t block = 0; block < num_blocks; ++block) {
            const int64_t matrix_offset = block * block_size * block_size;
            const int64_t vector_offset = block * block_size;

            for (int64_t out = 0; out < block_size; ++out) {
                float sum = 0.0f;
                for (int64_t in = 0; in < block_size; ++in) {
                    sum += after_right[perm[vector_offset + in]] *
                           llama_monarch_factor_at(left, matrix_offset + in * block_size + out);
                }
                y[vector_offset + out] = sum;
            }
        }
    }
}

ggml_tensor * llama_monarch_linear(
        ggml_context * ctx,
        ggml_tensor  * input,
        ggml_tensor  * left,
        ggml_tensor  * right,
        ggml_tensor  * permutation) {
    GGML_ASSERT(input != nullptr && left != nullptr && right != nullptr && permutation != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(left->type == GGML_TYPE_F32 || left->type == GGML_TYPE_F16);
    GGML_ASSERT(right->type == GGML_TYPE_F32 || right->type == GGML_TYPE_F16);
    GGML_ASSERT(permutation->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(left));
    GGML_ASSERT(ggml_is_contiguous(right));
    GGML_ASSERT(ggml_is_contiguous(permutation));

    GGML_ASSERT(left->ne[0] == left->ne[1]);
    GGML_ASSERT(left->ne[0] == right->ne[0]);
    GGML_ASSERT(left->ne[1] == right->ne[1]);
    GGML_ASSERT(left->ne[2] == right->ne[2]);
    GGML_ASSERT(left->ne[3] == 1 && right->ne[3] == 1);

    const int64_t width = left->ne[0] * left->ne[2];
    GGML_ASSERT(input->ne[0] == width);
    GGML_ASSERT(ggml_nelements(permutation) == width);

    ggml_tensor * args[] = { input, left, right, permutation };
    return ggml_custom_4d(
        ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 4, llama_monarch_compute, GGML_N_TASKS_MAX, nullptr);
}
