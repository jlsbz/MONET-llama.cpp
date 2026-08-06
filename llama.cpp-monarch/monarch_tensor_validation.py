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

_ARCHITECTURE_PROJECTIONS = {
    "llama": frozenset(_PROJECTION_NAMES),
    # Qwen2.5-7B uses GQA.  The current runtime deliberately keeps its
    # rectangular K/V projections dense and only claims square Q/O factors.
    "qwen2": frozenset(("q_proj", "o_proj")),
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
    """Map one supported Hugging Face attention layer to its GGUF base name."""
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


def validate_monarch_layer_for_architecture(
    layer_name: str,
    model_architecture: str,
    *,
    source: str | Path | None = None,
) -> None:
    """Reject factors that the selected llama.cpp architecture cannot claim."""
    prefix = _source_prefix(source)
    if not isinstance(model_architecture, str):
        raise TypeError(
            f"{prefix}model architecture must be a string, "
            f"got {type(model_architecture).__name__}"
        )

    allowed = _ARCHITECTURE_PROJECTIONS.get(model_architecture)
    if allowed is None:
        supported = ", ".join(sorted(_ARCHITECTURE_PROJECTIONS))
        raise ValueError(
            f"{prefix}Monarch runtime is not enabled for model architecture "
            f"{model_architecture!r}; supported architectures: {supported}"
        )

    match = _HF_LAYER_PATTERN.fullmatch(layer_name)
    if match is None:
        raise ValueError(f"{prefix}unsupported Monarch layer name: {layer_name}")

    projection = match.group(2)
    if projection not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"{prefix}{model_architecture} Monarch runtime does not support "
            f"{projection}; allowed projections: {allowed_text}"
        )


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

    validate_monarch_structure(
        left,
        right,
        permutation,
        source=source,
    )


def validate_monarch_metadata(
    obj: Mapping[str, Any],
    *,
    width: int,
    block_size: int,
    num_blocks: int,
    source: str | Path | None = None,
) -> None:
    """Cross-check optional artifact metadata against its actual arrays."""
    prefix = _source_prefix(source)

    def optional_int(name: str) -> int | None:
        if name not in obj:
            return None
        value = np.asarray(obj[name])
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{prefix}{name} must be one integer scalar")
        return int(value.reshape(-1)[0])

    in_features = optional_int("in_features")
    out_features = optional_int("out_features")
    saved_block_size = optional_int("block_size")
    saved_num_blocks = optional_int("num_blocks")

    if in_features is not None and in_features != width:
        raise ValueError(
            f"{prefix}in_features={in_features} does not match factor width {width}"
        )
    if out_features is not None and out_features != width:
        raise ValueError(
            f"{prefix}out_features={out_features} does not match square factor width {width}"
        )
    if saved_block_size is not None and saved_block_size != block_size:
        raise ValueError(
            f"{prefix}block_size={saved_block_size} does not match factor block size {block_size}"
        )
    if saved_num_blocks is not None and saved_num_blocks != num_blocks:
        raise ValueError(
            f"{prefix}num_blocks={saved_num_blocks} does not match factor count {num_blocks}"
        )

    # GGUF already carries the model's original projection bias. The current
    # runtime applies that bias after the Monarch OP, so separately fitted bias
    # values would be ambiguous and must not be silently discarded.
    if obj.get("bias") is not None:
        raise ValueError(
            f"{prefix}contains a fitted bias; refit weight-only factors and keep "
            "the original model bias in GGUF"
        )


def validate_monarch_structure(
    left: np.ndarray,
    right: np.ndarray,
    permutation: np.ndarray,
    *,
    source: str | Path | None = None,
) -> int:
    """Validate square Monarch factors and return their flattened width."""
    prefix = _source_prefix(source)

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

    return expected_width
