#pragma once

struct ggml_context;
struct ggml_tensor;

// Build the portable CPU reference Monarch linear operator used by MONET.
//
// Row-vector contract:
//     input -> blockdiag(R) -> gather(perm) -> blockdiag(L)
//
// input is contiguous F32. L/R are contiguous F32 or F16 tensors shaped
// [block_size, block_size, num_blocks] in GGML order. perm is contiguous I32.
ggml_tensor * llama_monarch_linear(
        ggml_context * ctx,
        ggml_tensor  * input,
        ggml_tensor  * left,
        ggml_tensor  * right,
        ggml_tensor  * permutation);
