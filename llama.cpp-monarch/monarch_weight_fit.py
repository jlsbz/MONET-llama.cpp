#!/usr/bin/env python3
"""NumPy helpers for a deterministic fixed-permutation Monarch baseline."""

from __future__ import annotations

import numpy as np


def fit_square_weight_rank1_blocks(
    weight: np.ndarray,
    *,
    block_size: int = 64,
    power_iterations: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Approximate one HF Linear weight for the fixed MONET permutation.

    ``weight`` uses HF/PyTorch layout [out_features, in_features]. For the
    special width ``block_size ** 2`` used by Llama-2-7B with block_size=64,
    every block pair in the row-vector dense transform is represented by one
    rank-1 outer product. Batched power iteration finds that product without
    requiring PyTorch or calibration activations.

    This is a deterministic weight-space baseline for conversion/runtime
    validation. It does not replace activation-aware fitting.
    """
    weight = np.asarray(weight)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"weight must be square rank-2; got {weight.shape}")
    if not np.issubdtype(weight.dtype, np.floating):
        raise TypeError(f"weight must use a floating dtype; got {weight.dtype}")
    if block_size <= 0 or weight.shape[0] % block_size != 0:
        raise ValueError("weight width must be divisible by a positive block_size")
    if power_iterations <= 0:
        raise ValueError("power_iterations must be positive")

    width = weight.shape[0]
    num_blocks = width // block_size
    if num_blocks != block_size:
        raise ValueError(
            "the analytic fixed-permutation baseline currently requires "
            "num_blocks == block_size"
        )

    # HF uses y = x @ weight.T. Partition that row-vector transform into
    # [source_block, destination_block, source_coordinate, destination_coordinate].
    dense = weight.astype(np.float32, copy=False).T
    blocks = dense.reshape(num_blocks, block_size, num_blocks, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)

    rng = np.random.default_rng(seed)
    right_vectors = rng.standard_normal(
        (num_blocks, num_blocks, block_size), dtype=np.float32
    )
    right_vectors /= np.maximum(
        np.linalg.norm(right_vectors, axis=-1, keepdims=True), 1e-20
    )

    for _ in range(power_iterations):
        left_vectors = np.einsum(
            "sdij,sdj->sdi", blocks, right_vectors, optimize=True
        )
        left_vectors /= np.maximum(
            np.linalg.norm(left_vectors, axis=-1, keepdims=True), 1e-20
        )
        right_vectors = np.einsum(
            "sdij,sdi->sdj", blocks, left_vectors, optimize=True
        )
        right_vectors /= np.maximum(
            np.linalg.norm(right_vectors, axis=-1, keepdims=True), 1e-20
        )

    left_vectors = np.einsum(
        "sdij,sdj->sdi", blocks, right_vectors, optimize=True
    )
    singular_values = np.linalg.norm(left_vectors, axis=-1)
    left_vectors /= np.maximum(singular_values[..., None], 1e-20)

    scale = np.sqrt(singular_values, dtype=np.float32)
    # R[source_block, source_coordinate, destination_block]
    right_factors = (left_vectors * scale[..., None]).transpose(0, 2, 1)
    # L[destination_block, source_block, destination_coordinate]
    left_factors = (right_vectors * scale[..., None]).transpose(1, 0, 2)

    permutation = np.arange(width, dtype=np.int32)
    permutation = permutation.reshape(num_blocks, block_size).T.reshape(-1)

    approximation = np.einsum(
        "sdi,sdj->sdij",
        left_vectors * singular_values[..., None],
        right_vectors,
        optimize=True,
    )
    squared_error = float(np.sum((blocks - approximation) ** 2, dtype=np.float64))
    squared_reference = float(np.sum(blocks ** 2, dtype=np.float64))
    relative_squared_error = squared_error / max(squared_reference, 1e-30)

    metrics = {
        "squared_error": squared_error,
        "squared_reference": squared_reference,
        "relative_frobenius_squared_error": relative_squared_error,
        "relative_frobenius_error": float(np.sqrt(relative_squared_error)),
        "captured_energy_ratio": 1.0 - relative_squared_error,
    }
    return (
        left_factors.astype(np.float32, copy=False),
        right_factors.astype(np.float32, copy=False),
        permutation,
        metrics,
    )
