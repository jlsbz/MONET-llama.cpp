#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from monarch_tensor_validation import validate_monarch_structure


def _validated_reference_inputs(
    inputs: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.dtype]:
    left = np.asarray(left)
    right = np.asarray(right)
    permutation = np.asarray(permutation)
    width = validate_monarch_structure(left, right, permutation)

    inputs = np.asarray(inputs)
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if inputs.shape[-1] != width:
        raise ValueError(
            f"inputs last dimension must equal Monarch width ({width}); "
            f"got shape {inputs.shape}"
        )
    if not np.issubdtype(inputs.dtype, np.floating):
        raise TypeError(f"inputs must use a floating dtype; got {inputs.dtype}")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs contain NaN or infinite values")

    num_blocks, block_size, _ = left.shape
    compute_dtype = np.dtype(
        np.result_type(inputs.dtype, left.dtype, right.dtype, np.float32)
    )

    return (
        inputs.astype(compute_dtype, copy=False),
        left.astype(compute_dtype, copy=False),
        right.astype(compute_dtype, copy=False),
        permutation.astype(np.intp, copy=False),
        num_blocks,
        block_size,
        compute_dtype,
    )


def monarch_linear_reference(
    inputs: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """
    Apply the square Monarch transform used by the fitting prototype.

    The row-vector contract is:
        inputs -> blockdiag(R) -> gather(perm) -> blockdiag(L)
    """
    (
        inputs,
        left,
        right,
        permutation,
        num_blocks,
        block_size,
        _,
    ) = _validated_reference_inputs(inputs, left, right, permutation)

    original_shape = inputs.shape
    blocked = inputs.reshape(*original_shape[:-1], num_blocks, block_size)
    after_right = np.einsum("...bi,bij->...bj", blocked, right, optimize=True)

    flattened = after_right.reshape(*original_shape[:-1], -1)
    permuted = np.take(flattened, permutation, axis=-1)

    blocked = permuted.reshape(*original_shape[:-1], num_blocks, block_size)
    after_left = np.einsum("...bi,bij->...bj", blocked, left, optimize=True)
    return after_left.reshape(original_shape)


def materialize_monarch_dense_reference(
    left: np.ndarray,
    right: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """
    Materialize the same transform as a dense row-vector matrix.

    This deliberately uses explicit block placement and a permutation matrix so
    tests can compare it with ``monarch_linear_reference`` without sharing the
    forward implementation.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    permutation = np.asarray(permutation)
    width = validate_monarch_structure(left, right, permutation)

    num_blocks, block_size, _ = left.shape
    compute_dtype = np.dtype(np.result_type(left.dtype, right.dtype, np.float32))
    dense_left = np.zeros((width, width), dtype=compute_dtype)
    dense_right = np.zeros((width, width), dtype=compute_dtype)

    for block_index in range(num_blocks):
        start = block_index * block_size
        stop = start + block_size
        dense_left[start:stop, start:stop] = left[block_index]
        dense_right[start:stop, start:stop] = right[block_index]

    permutation_matrix = np.zeros((width, width), dtype=compute_dtype)
    columns = np.arange(width, dtype=np.intp)
    permutation_matrix[permutation.astype(np.intp, copy=False), columns] = 1

    return dense_right @ permutation_matrix @ dense_left
