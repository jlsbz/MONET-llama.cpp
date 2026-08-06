#!/usr/bin/env python3
"""Create a tiny deterministic LLaMA/Qwen2 GGUF for MONET integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "llama.cpp-monarch" / "gguf-py"))

import gguf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--architecture",
        choices=("llama", "qwen2"),
        default="llama",
        help="GGUF architecture; qwen2 uses a tiny GQA geometry",
    )
    parser.add_argument(
        "--monarch-projections",
        choices=("q", "qo", "none"),
        default=None,
        help="defaults to q for llama and qo for qwen2",
    )
    parser.add_argument(
        "--incomplete-monarch",
        action="store_true",
        help="write only monarch_l to exercise strict incomplete-triple rejection",
    )
    parser.add_argument(
        "--unsupported-k-monarch",
        action="store_true",
        help="add a K triple that the square-only Qwen2 GQA loader must reject",
    )
    args = parser.parse_args()

    monarch_projections = args.monarch_projections
    if monarch_projections is None:
        monarch_projections = "qo" if args.architecture == "qwen2" else "q"
    if args.incomplete_monarch and monarch_projections == "none":
        parser.error("--incomplete-monarch requires a Q Monarch projection")
    if args.unsupported_k_monarch and args.architecture != "qwen2":
        parser.error("--unsupported-k-monarch is only valid for --architecture qwen2")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_vocab = 128
    n_ctx = 128
    n_embd = 64
    n_head = 4 if args.architecture == "qwen2" else 1
    n_head_kv = 2 if args.architecture == "qwen2" else n_head
    n_embd_gqa = n_embd * n_head_kv // n_head
    n_ff = 128
    n_layer = 1

    writer = gguf.GGUFWriter(args.output, args.architecture)
    writer.add_name(f"MONET synthetic {args.architecture} Monarch loader fixture")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_vocab_size(n_vocab)
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("no_vocab")

    rng = np.random.default_rng(20260806)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) * 0.01).astype(np.float16)

    writer.add_tensor("token_embd.weight", matrix(n_vocab, n_embd))
    writer.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("output.weight", matrix(n_vocab, n_embd))

    writer.add_tensor("blk.0.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", matrix(n_embd, n_embd))
    writer.add_tensor("blk.0.attn_k.weight", matrix(n_embd_gqa, n_embd))
    writer.add_tensor("blk.0.attn_v.weight", matrix(n_embd_gqa, n_embd))
    writer.add_tensor("blk.0.attn_output.weight", matrix(n_embd, n_embd))

    if args.architecture == "qwen2":
        # Qwen2 uses Q/K/V bias.  Non-zero values exercise the existing rule:
        # the bias is added after the Monarch Q projection, not fitted twice.
        writer.add_tensor("blk.0.attn_q.bias", np.full(n_embd, 0.001, dtype=np.float32))
        writer.add_tensor("blk.0.attn_k.bias", np.full(n_embd_gqa, 0.002, dtype=np.float32))
        writer.add_tensor("blk.0.attn_v.bias", np.full(n_embd_gqa, 0.003, dtype=np.float32))

    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate.weight", matrix(n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_up.weight", matrix(n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_down.weight", matrix(n_embd, n_ff))

    identity = np.eye(n_embd, dtype=np.float16)[None, :, :]

    def add_monarch(projection: str, *, incomplete: bool = False) -> None:
        writer.add_tensor(f"blk.0.{projection}.monarch_l", identity)
        if not incomplete:
            writer.add_tensor(f"blk.0.{projection}.monarch_r", identity)
            writer.add_tensor(
                f"blk.0.{projection}.monarch_perm",
                np.arange(n_embd, dtype=np.int32),
            )

    if monarch_projections in {"q", "qo"}:
        add_monarch("attn_q", incomplete=args.incomplete_monarch)
    if monarch_projections == "qo":
        add_monarch("attn_output")
    if args.unsupported_k_monarch:
        add_monarch("attn_k")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(args.output)


if __name__ == "__main__":
    main()
