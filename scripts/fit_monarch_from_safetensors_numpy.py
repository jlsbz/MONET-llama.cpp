#!/usr/bin/env python3
"""Create a real-model MONET conversion artifact without PyTorch.

This utility reads selected tensors directly from a local safetensors index and
uses a deterministic weight-space rank-1 block baseline. It is intended to
validate steps 6-9 when a full activation-aware fitting environment is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
LLAMA_DIR = REPOSITORY_DIR / "llama.cpp-monarch"
sys.path.insert(0, str(LLAMA_DIR))

from monarch_weight_fit import fit_square_weight_rank1_blocks  # noqa: E402


SAFETENSORS_DTYPES = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}


def read_safetensors_tensor(path: Path, tensor_name: str) -> np.ndarray:
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length).decode("utf-8"))

    if tensor_name not in header:
        raise KeyError(f"{path}: tensor not found: {tensor_name}")
    entry = header[tensor_name]
    dtype_name = entry["dtype"]
    if dtype_name not in SAFETENSORS_DTYPES:
        raise TypeError(f"{path}: unsupported safetensors dtype {dtype_name}")

    start, end = entry["data_offsets"]
    dtype = SAFETENSORS_DTYPES[dtype_name]
    expected_bytes = int(np.prod(entry["shape"], dtype=np.int64)) * dtype.itemsize
    if end - start != expected_bytes:
        raise ValueError(f"{path}: inconsistent byte count for {tensor_name}")

    data_offset = 8 + header_length + start
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=tuple(entry["shape"]),
        order="C",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="local Hugging Face model directory")
    parser.add_argument("output", type=Path, help="new directory for .npz artifacts")
    parser.add_argument(
        "--targets",
        default="model.layers.0.self_attn.q_proj",
        help="comma-separated module names without the trailing .weight",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index_path = args.model / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    targets = [name.strip() for name in args.targets.split(",") if name.strip()]
    for target_index, layer_name in enumerate(targets):
        tensor_name = f"{layer_name}.weight"
        if tensor_name not in weight_map:
            raise KeyError(f"model index does not contain {tensor_name}")

        shard_path = args.model / weight_map[tensor_name]
        print(f"[MONET] reading {tensor_name} from {shard_path}")
        weight = read_safetensors_tensor(shard_path, tensor_name)
        left, right, permutation, metrics = fit_square_weight_rank1_blocks(
            weight,
            block_size=args.block_size,
            power_iterations=args.power_iterations,
            seed=args.seed + target_index,
        )

        output_name = layer_name.replace(".", "__").replace("/", "_") + ".npz"
        output_path = args.output / output_name
        np.savez(
            output_path,
            layer_name=np.asarray(layer_name),
            in_features=np.asarray(weight.shape[1], dtype=np.int64),
            out_features=np.asarray(weight.shape[0], dtype=np.int64),
            block_size=np.asarray(args.block_size, dtype=np.int64),
            num_blocks=np.asarray(weight.shape[0] // args.block_size, dtype=np.int64),
            L=left,
            R=right,
            perm=permutation,
            relative_frobenius_squared_error=np.asarray(
                metrics["relative_frobenius_squared_error"], dtype=np.float64
            ),
        )
        print(f"[MONET] wrote {output_path}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        summaries.append({"layer_name": layer_name, "artifact": output_name, **metrics})

    summary_path = args.output / "fit_summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    print(f"[MONET] wrote {summary_path}")


if __name__ == "__main__":
    main()
