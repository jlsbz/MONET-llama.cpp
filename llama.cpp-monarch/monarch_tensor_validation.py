#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import numpy as np


_HF_LAYER_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)

_PROJECTION_NAMES = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
}


def _source_prefix(source: str | Path | None) -> str:
    return f"{source}: " if source is not None else ""


def extract_monarch_fields(
    obj: object,
    *,
    source: str | Path | None = None,
) -> tuple[str, Any, Any, Any]:
    """Extract the normalized layer name and required tensors from a saved object."""
    prefix = _source_prefix(source)

    if not isinstance(obj, Mapping):
        raise TypeError(f"{prefix}expected a mapping, got {type(obj).__name__}")

    layer_name = obj.get("layer_name", obj.get("target_module_name"))
    if layer_name is None:
        raise KeyError(
            f"{prefix}does not contain 'layer_name' or 'target_module_name'"
        )
    if not isinstance(layer_name, str):
        raise TypeError(
            f"{prefix}layer name must be a string, got {type(layer_name).__name__}"
        )
    if not layer_name.strip():
        raise ValueError(f"{prefix}layer name must not be empty")

    missing = [name for name in ("L", "R", "perm") if name not in obj]
    if missing:
        quoted = ", ".join(repr(name) for name in missing)
        raise KeyError(f"{prefix}does not contain required key(s): {quoted}")

    return layer_name, obj["L"], obj["R"], obj["perm"]


def hf_layer_to_gguf_base(layer_name: str) -> str | None:
    """Map one supported Hugging Face LLaMA attention layer to its GGUF base name."""
    if not isinstance(layer_name, str):
        raise TypeError(
            f"layer name must be a string, got {type(layer_name).__name__}"
        )

    match = _HF_LAYER_PATTERN.fullmatch(layer_name)
    if match is None:
        return None

    layer_id = int(match.group(1))
    projection = _PROJECTION_NAMES[match.group(2)]
    return f"blk.{layer_id}.{projection}"


def validate_monarch_arrays(
    layer_name: str,
    left: np.ndarray,
    right: np.ndarray,
    permutation: np.ndarray,
    *,
    source: str | Path | None = None,
) -> None:
    """Validate the block factors and permutation expected by the current loader."""
    prefix = _source_prefix(source)

    if hf_layer_to_gguf_base(layer_name) is None:
        raise ValueError(f"{prefix}unsupported Monarch layer name: {layer_name}")

    left = np.asarray(left)
    right = np.asarray(right)
    permutation = np.asarray(permutation)

    if left.ndim != 3 or right.ndim != 3:
        raise ValueError(
            f"{prefix}L and R must both be rank-3 [num_blocks, block_size, "
            f"block_size] tensors; got L{left.shape} and R{right.shape}"
        )
    if left.shape != right.shape:
        raise ValueError(
            f"{prefix}L and R shapes must match; got L{left.shape} and R{right.shape}"
        )
    if any(dimension <= 0 for dimension in left.shape):
        raise ValueError(f"{prefix}L and R dimensions must be positive; got {left.shape}")

    num_blocks, left_rows, left_columns = left.shape
    if left_rows != left_columns:
        raise ValueError(
            f"{prefix}each Monarch block must be square; got block shape "
            f"({left_rows}, {left_columns})"
        )

    if not np.issubdtype(left.dtype, np.floating):
        raise TypeError(f"{prefix}L must use a floating dtype; got {left.dtype}")
    if not np.issubdtype(right.dtype, np.floating):
        raise TypeError(f"{prefix}R must use a floating dtype; got {right.dtype}")
    if not np.isfinite(left).all():
        raise ValueError(f"{prefix}L contains NaN or infinite values")
    if not np.isfinite(right).all():
        raise ValueError(f"{prefix}R contains NaN or infinite values")

    expected_width = num_blocks * left_rows
    if permutation.ndim != 1:
        raise ValueError(
            f"{prefix}perm must be rank-1 with {expected_width} entries; "
            f"got shape {permutation.shape}"
        )
    if permutation.shape[0] != expected_width:
        raise ValueError(
            f"{prefix}perm length must equal num_blocks * block_size "
            f"({expected_width}); got {permutation.shape[0]}"
        )
    if not np.issubdtype(permutation.dtype, np.integer):
        raise TypeError(
            f"{prefix}perm must use an integer dtype; got {permutation.dtype}"
        )

    expected_permutation = np.arange(expected_width, dtype=np.int64)
    normalized_permutation = np.sort(permutation.astype(np.int64, copy=False))
    if not np.array_equal(normalized_permutation, expected_permutation):
        raise ValueError(
            f"{prefix}perm must contain each index in [0, {expected_width}) exactly once"
        )
