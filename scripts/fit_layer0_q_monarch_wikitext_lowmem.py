#!/usr/bin/env python3
"""Low-memory activation-aware MONET fit for LLaMA layer-0 Q projection.

This recovery utility avoids loading the full causal LM.  For layer 0, the
q_proj input can be reconstructed exactly from token embeddings followed by
the first input RMSNorm.  Only the required safetensors are mmap'ed.

It is intended to validate a local WikiText-2 calibration path under tight
host-memory constraints.  Deeper layers still require a real sequential/full
model forward and are outside this script's contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
LLAMA_DIR = REPOSITORY_DIR / "llama.cpp-monarch"
sys.path.insert(0, str(LLAMA_DIR))

from monarch_weight_fit import fit_square_weight_rank1_blocks  # noqa: E402
from monarch_tensor_validation import validate_monarch_arrays  # noqa: E402
from fit_monarch_from_safetensors_numpy import read_safetensors_tensor  # noqa: E402


LAYER_NAME = "model.layers.0.self_attn.q_proj"
EMBEDDING_TENSOR = "model.embed_tokens.weight"
NORM_TENSOR = "model.layers.0.input_layernorm.weight"
Q_WEIGHT_TENSOR = f"{LAYER_NAME}.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def find_tensor_file(model_path: Path, tensor_name: str) -> Path:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(f"model index does not contain {tensor_name}")
    return model_path / weight_map[tensor_name]


def load_texts(dataset_path: Path, split: str, max_samples: int, min_chars: int) -> tuple[list[str], list[Path]]:
    if dataset_path.is_file():
        files = [dataset_path]
    else:
        files = sorted(dataset_path.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files found under {dataset_path}")

    texts: list[str] = []
    for path in files:
        table = parquet.read_table(path, columns=["text"])
        for value in table.column("text").to_pylist():
            if value is None:
                continue
            text = value.strip()
            if len(text) < min_chars:
                continue
            if text.startswith("=") and text.endswith("="):
                continue
            texts.append(text)
            if len(texts) >= max_samples:
                return texts, files
    return texts, files


class MonarchLinear(nn.Module):
    def __init__(self, left: np.ndarray, right: np.ndarray, permutation: np.ndarray):
        super().__init__()
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError("left/right must have matching rank-3 shapes")
        self.num_blocks, self.block_size, block_size_2 = left.shape
        if self.block_size != block_size_2:
            raise ValueError("factor blocks must be square")
        self.width = self.num_blocks * self.block_size
        if permutation.shape != (self.width,):
            raise ValueError("permutation width does not match factors")

        self.L = nn.Parameter(torch.from_numpy(np.array(left, dtype=np.float32, copy=True)))
        self.R = nn.Parameter(torch.from_numpy(np.array(right, dtype=np.float32, copy=True)))
        self.register_buffer(
            "perm",
            torch.from_numpy(np.array(permutation, dtype=np.int64, copy=True)),
            persistent=True,
        )

    def block_diag_mul(self, inputs: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        rows = inputs.shape[0]
        inputs = inputs.reshape(rows, self.num_blocks, self.block_size)
        return torch.einsum("tbi,bij->tbj", inputs, blocks).reshape(rows, self.width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block_diag_mul(inputs.float(), self.R.float())
        outputs = outputs[:, self.perm]
        return self.block_diag_mul(outputs, self.L.float())


@torch.no_grad()
def activation_metrics(
    monarch: MonarchLinear,
    activations: torch.Tensor,
    reference_weight: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    total_squared_error = 0.0
    total_squared_reference = 0.0
    total_absolute_error = 0.0
    total_absolute_reference = 0.0
    total_elements = 0

    for start in range(0, activations.shape[0], batch_size):
        inputs = activations[start : start + batch_size]
        reference = F.linear(inputs, reference_weight, bias=None)
        approximate = monarch(inputs)
        difference = approximate - reference
        total_squared_error += difference.square().sum().item()
        total_squared_reference += reference.square().sum().item()
        total_absolute_error += difference.abs().sum().item()
        total_absolute_reference += reference.abs().sum().item()
        total_elements += reference.numel()

    return {
        "mse": total_squared_error / max(total_elements, 1),
        "relative_mse": total_squared_error / max(total_squared_reference, 1e-30),
        "mae": total_absolute_error / max(total_elements, 1),
        "relative_mae": total_absolute_error / max(total_absolute_reference, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--initial-artifact",
        type=Path,
        default=None,
        help="Optional existing .npz factors to fine-tune or evaluate with --steps 0.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.steps < 0 or args.batch_size <= 0 or args.num_samples <= 0:
        raise ValueError("steps must be non-negative; batch size and sample count must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    with (args.model / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    hidden_size = int(config["hidden_size"])
    rms_norm_eps = float(config["rms_norm_eps"])

    texts, dataset_files = load_texts(args.dataset, args.split, args.num_samples, args.min_chars)
    if not texts:
        raise RuntimeError("no usable calibration texts were found")
    print(f"[Dataset] selected {len(texts)} texts from {len(dataset_files)} Parquet file(s)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        local_files_only=True,
    )
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )
    token_ids = np.concatenate(
        [np.asarray(row, dtype=np.int64) for row in encoded["input_ids"]]
    )
    if token_ids.size == 0:
        raise RuntimeError("tokenizer produced no calibration tokens")
    print(f"[Dataset] calibration tokens: {token_ids.size}")

    embedding_path = find_tensor_file(args.model, EMBEDDING_TENSOR)
    norm_path = find_tensor_file(args.model, NORM_TENSOR)
    q_weight_path = find_tensor_file(args.model, Q_WEIGHT_TENSOR)

    embeddings = read_safetensors_tensor(embedding_path, EMBEDDING_TENSOR)
    norm_weight = np.asarray(
        read_safetensors_tensor(norm_path, NORM_TENSOR), dtype=np.float32
    )
    if embeddings.shape[1] != hidden_size or norm_weight.shape != (hidden_size,):
        raise ValueError("model embedding/RMSNorm shapes do not match config.hidden_size")

    # Layer-0 q_proj input: token embedding followed by layer-0 input RMSNorm.
    activations_np = np.asarray(embeddings[token_ids], dtype=np.float32)
    variance = np.mean(np.square(activations_np), axis=-1, keepdims=True)
    activations_np *= 1.0 / np.sqrt(variance + rms_norm_eps)
    activations_np *= norm_weight
    activations = torch.from_numpy(np.ascontiguousarray(activations_np))
    del activations_np, embeddings

    q_weight_np = read_safetensors_tensor(q_weight_path, Q_WEIGHT_TENSOR)
    if q_weight_np.shape != (hidden_size, hidden_size):
        raise ValueError(f"unexpected q_proj shape: {q_weight_np.shape}")
    if args.initial_artifact is None:
        left, right, permutation, weight_metrics = fit_square_weight_rank1_blocks(
            q_weight_np,
            block_size=args.block_size,
            power_iterations=args.power_iterations,
            seed=args.seed,
        )
        initialization = "deterministic weight-space rank-1 block baseline"
    else:
        with np.load(args.initial_artifact, allow_pickle=False) as archive:
            artifact_layer_name = str(np.asarray(archive["layer_name"]).item())
            left = np.asarray(archive["L"])
            right = np.asarray(archive["R"])
            permutation = np.asarray(archive["perm"])
        if artifact_layer_name != LAYER_NAME:
            raise ValueError(
                f"initial artifact targets {artifact_layer_name}, expected {LAYER_NAME}"
            )
        validate_monarch_arrays(
            artifact_layer_name,
            left,
            right,
            permutation,
            source=args.initial_artifact,
        )
        weight_metrics = None
        initialization = f"existing artifact: {args.initial_artifact.resolve()}"
    reference_weight = torch.from_numpy(
        np.ascontiguousarray(np.asarray(q_weight_np, dtype=np.float32))
    )
    del q_weight_np

    monarch = MonarchLinear(left, right, permutation)
    initial_metrics = activation_metrics(
        monarch, activations, reference_weight, args.eval_batch_size
    )
    print(f"[Initial] relative_mse={initial_metrics['relative_mse']:.8f}")

    optimizer = torch.optim.AdamW(
        monarch.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_relative_mse = initial_metrics["relative_mse"]
    best_left = monarch.L.detach().clone()
    best_right = monarch.R.detach().clone()

    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0,
            activations.shape[0],
            (min(args.batch_size, activations.shape[0]),),
            generator=generator,
        )
        inputs = activations[indices]
        with torch.no_grad():
            reference = F.linear(inputs, reference_weight, bias=None)
        approximate = monarch(inputs)
        mse = F.mse_loss(approximate, reference)
        relative_mse = mse / reference.square().mean().clamp_min(1e-30)

        optimizer.zero_grad(set_to_none=True)
        mse.backward()
        torch.nn.utils.clip_grad_norm_(monarch.parameters(), 1.0)
        optimizer.step()

        relative_value = float(relative_mse.detach())
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            checkpoint_metrics = activation_metrics(
                monarch, activations, reference_weight, args.eval_batch_size
            )
            checkpoint_relative_mse = checkpoint_metrics["relative_mse"]
            if checkpoint_relative_mse < best_relative_mse:
                best_relative_mse = checkpoint_relative_mse
                best_left = monarch.L.detach().clone()
                best_right = monarch.R.detach().clone()
            print(
                f"[Fit] step={step} mse={float(mse.detach()):.8e} "
                f"minibatch_relative_mse={relative_value:.8f} "
                f"full_relative_mse={checkpoint_relative_mse:.8f}"
            )

    with torch.no_grad():
        monarch.L.copy_(best_left)
        monarch.R.copy_(best_right)
    final_metrics = activation_metrics(
        monarch, activations, reference_weight, args.eval_batch_size
    )
    print(f"[Final] relative_mse={final_metrics['relative_mse']:.8f}")

    artifact_name = LAYER_NAME.replace(".", "__") + ".npz"
    artifact_path = args.output / artifact_name
    np.savez(
        artifact_path,
        layer_name=np.asarray(LAYER_NAME),
        in_features=np.asarray(hidden_size, dtype=np.int64),
        out_features=np.asarray(hidden_size, dtype=np.int64),
        block_size=np.asarray(args.block_size, dtype=np.int64),
        num_blocks=np.asarray(hidden_size // args.block_size, dtype=np.int64),
        L=monarch.L.detach().numpy(),
        R=monarch.R.detach().numpy(),
        perm=monarch.perm.detach().numpy().astype(np.int32),
        relative_mse=np.asarray(final_metrics["relative_mse"], dtype=np.float64),
    )

    summary = {
        "scope": "low-memory exact layer-0 activation reconstruction; q_proj only",
        "layer_name": LAYER_NAME,
        "model_path": str(args.model.resolve()),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in dataset_files
        ],
        "split": args.split,
        "num_texts": len(texts),
        "num_tokens": int(token_ids.size),
        "token_ids_sha256": hashlib.sha256(token_ids.tobytes()).hexdigest().upper(),
        "hidden_size": hidden_size,
        "rms_norm_eps": rms_norm_eps,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initialization": initialization,
        "weight_space_initialization": weight_metrics,
        "initial_activation_metrics": initial_metrics,
        "final_activation_metrics": final_metrics,
        "best_checkpoint_relative_mse": best_relative_mse,
        "artifact": artifact_name,
        "artifact_sha256": sha256_file(artifact_path),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    summary_path = args.output / "fit_summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[Output] {artifact_path}")
    print(f"[Output] {summary_path}")


if __name__ == "__main__":
    main()
