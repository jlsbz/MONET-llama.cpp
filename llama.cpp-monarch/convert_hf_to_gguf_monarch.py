#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np

import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
    _mistral_common_installed,
    _mistral_import_error_msg,
)
from monarch_tensor_validation import (
    extract_monarch_fields,
    hf_layer_to_gguf_base,
    validate_monarch_arrays,
    validate_monarch_layer_for_architecture,
    validate_monarch_metadata,
)


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


# Add Change!
# For Monarch tensor support

def safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "_")


def list_monarch_artifact_files(monarch_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, names in os.walk(monarch_dir):
        for name in names:
            if name.endswith((".pt", ".npz")):
                files.append(Path(root) / name)
    return sorted(files)


def load_monarch_obj(artifact_path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if artifact_path.suffix == ".npz":
        with np.load(artifact_path, allow_pickle=False) as archive:
            obj = {name: archive[name] for name in archive.files}
        if "layer_name" in obj:
            obj["layer_name"] = str(np.asarray(obj["layer_name"]).item())
    else:
        obj = torch.load(artifact_path, map_location="cpu", weights_only=True)

    layer_name, left, right, permutation = extract_monarch_fields(
        obj,
        source=artifact_path,
    )

    if artifact_path.suffix == ".pt":
        for tensor_name, tensor in (("L", left), ("R", right), ("perm", permutation)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{artifact_path}: {tensor_name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
        left = left.detach().cpu().numpy()
        right = right.detach().cpu().numpy()
        permutation = permutation.detach().cpu().numpy()

    left_np = np.asarray(left)
    right_np = np.asarray(right)
    perm_np = np.asarray(permutation)

    validate_monarch_arrays(
        layer_name, left_np, right_np, perm_np, source=artifact_path
    )
    num_blocks, block_size, _ = left_np.shape
    width = num_blocks * block_size
    validate_monarch_metadata(
        obj,
        width=width,
        block_size=block_size,
        num_blocks=num_blocks,
        source=artifact_path,
    )

    if block_size != 64:
        raise ValueError(
            f"{artifact_path}: current llama.cpp loader requires block_size=64; "
            f"got {block_size}"
        )

    return layer_name, left_np, right_np, perm_np


def load_all_monarch_tensors(
    monarch_dir: Path,
    dtype: str = "f32",
    model_architecture: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Load all fitted Monarch .pt/.npz files and convert them to GGUF entries.

    Output tensor names:
        blk.i.attn_q.monarch_l
        blk.i.attn_q.monarch_r
        blk.i.attn_q.monarch_perm

    We store L/R as F32 by default for debugging.
    perm is stored as int32.
    """
    if dtype not in {"f16", "f32"}:
        raise ValueError(f"Unsupported Monarch dtype: {dtype}")

    artifact_files = list_monarch_artifact_files(monarch_dir)

    if len(artifact_files) == 0:
        raise FileNotFoundError(f"No .pt or .npz files found in Monarch dir: {monarch_dir}")

    out: list[tuple[str, np.ndarray]] = []

    ok = 0
    skipped = 0
    seen_bases: dict[str, Path] = {}

    for artifact_path in artifact_files:
        layer_name, L, R, perm = load_monarch_obj(artifact_path)

        if model_architecture is not None:
            validate_monarch_layer_for_architecture(
                layer_name,
                model_architecture,
                source=artifact_path,
            )

        base = hf_layer_to_gguf_base(layer_name)
        if base is None:
            skipped += 1
            logger.warning(f"[Monarch] Skip unsupported layer name: {layer_name}")
            continue

        previous_path = seen_bases.get(base)
        if previous_path is not None:
            raise ValueError(
                f"{artifact_path}: duplicate Monarch layer {layer_name!r}; "
                f"{base!r} was already provided by {previous_path}"
            )
        seen_bases[base] = artifact_path

        if dtype == "f16":
            L_np = L.astype(np.float16, copy=False)
            R_np = R.astype(np.float16, copy=False)
        else:
            L_np = L.astype(np.float32, copy=False)
            R_np = R.astype(np.float32, copy=False)

        perm_np = perm.astype(np.int32, copy=False)

        out.append((f"{base}.monarch_l", L_np))
        out.append((f"{base}.monarch_r", R_np))
        out.append((f"{base}.monarch_perm", perm_np))

        ok += 1

    if ok == 0:
        raise ValueError(
            f"No supported Monarch attention layers found in: {monarch_dir}"
        )

    logger.info(
        f"[Monarch] Loaded {ok} Monarch layers, skipped {skipped}, "
        f"extra GGUF tensors = {len(out)}"
    )

    return out

def add_monarch_tensors_to_writer(model_instance, monarch_dir: Path, monarch_dtype: str = "f32") -> None:
    """
    Add Monarch tensors to model_instance.gguf_writer before model_instance.write().
        create model_instance
        add Monarch tensors to gguf_writer
        call model_instance.write()
    """
    if monarch_dir is None:
        return

    monarch_dir = Path(monarch_dir)

    if not monarch_dir.is_dir():
        raise FileNotFoundError(f"Monarch dir does not exist: {monarch_dir}")

    if not hasattr(model_instance, "gguf_writer"):
        raise AttributeError(
            "model_instance does not have gguf_writer. Cannot add Monarch tensors."
        )

    writer = model_instance.gguf_writer

    model_arch = getattr(model_instance, "model_arch", None)
    try:
        model_architecture = gguf.MODEL_ARCH_NAMES[model_arch]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot determine GGUF model architecture for Monarch tensors: {model_arch!r}"
        ) from exc

    logger.info("[Monarch] Adding extra Monarch tensors to GGUF writer before model_instance.write()...")
    extra_tensors = load_all_monarch_tensors(
        monarch_dir,
        dtype=monarch_dtype,
        model_architecture=model_architecture,
    )

    for name, arr in extra_tensors:
        logger.info(f"[Monarch] Add tensor: {name}, shape={arr.shape}, dtype={arr.dtype}")
        writer.add_tensor(name, arr)

    logger.info(f"[Monarch] Added {len(extra_tensors)} extra tensors before writing GGUF.")


# def install_monarch_tensor_patch(model_instance, monarch_dir: Path, monarch_dtype: str = "f32") -> None:
#     """
#     Monkey patch model_instance.write_tensors().

#     Original flow:
#         model_instance.write()
#             -> model_instance.write_tensors()
#             -> GGUF writer writes model tensors

#     Patched flow:
#         model_instance.write()
#             -> original write_tensors()
#             -> add Monarch L/R/perm tensors to same GGUF writer
#     """
#     if monarch_dir is None:
#         return

#     monarch_dir = Path(monarch_dir)

#     if not monarch_dir.is_dir():
#         raise FileNotFoundError(f"Monarch dir does not exist: {monarch_dir}")

#     if not hasattr(model_instance, "write_tensors"):
#         raise AttributeError(
#             "model_instance does not have write_tensors(). "
#             "Your llama.cpp conversion API may have changed. "
#             "In that case, add the Monarch tensor insertion at the end of the model class's tensor-writing method."
#         )

#     if not hasattr(model_instance, "gguf_writer"):
#         raise AttributeError(
#             "model_instance does not have gguf_writer. "
#             "Cannot add Monarch tensors."
#         )

#     original_write_tensors = model_instance.write_tensors

#     def patched_write_tensors(self, *args, **kwargs):
#         logger.info("[Monarch] Calling original write_tensors()...")
#         ret = original_write_tensors(*args, **kwargs)

#         logger.info("[Monarch] Adding extra Monarch tensors...")
#         extra_tensors = load_all_monarch_tensors(monarch_dir, dtype=monarch_dtype)

#         for name, arr in extra_tensors:
#             logger.info(f"[Monarch] Add tensor: {name}, shape={arr.shape}, dtype={arr.dtype}")
#             self.gguf_writer.add_tensor(name, arr)

#         logger.info(f"[Monarch] Added {len(extra_tensors)} extra tensors.")
#         return ret

#     model_instance.write_tensors = MethodType(patched_write_tensors, model_instance)

# Change end!

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="auto",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split (disabled by default)"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="Print the supported models"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj", action="store_true",
        help="Export multimodal projector (mmproj) for vision models. This will only work on some vision models. An 'mmproj-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--mtp", action="store_true",
        help="Export only the multi-token prediction (MTP) head as a separate GGUF, suitable for use as a speculative draft. An 'mtp-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="Exclude the multi-token prediction (MTP) head from the converted GGUF. Pair with --mtp on a second run to publish trunk and MTP as two files. Note: the split form duplicates embeddings, but even though the bundled default is more space-efficient overall, this allows differing quantization which may be more performant.",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="Whether the model is stored following the Mistral format.",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help=(
            "Whether to disable usage of Mistral community chat templates. If set, use the Mistral official `mistral-common` library for tokenization and detokenization of Mistral models. "
            "Using `mistral-common` ensure correctness and zero-day support of tokenization for models converted from the Mistral format but requires to manually setup the tokenization server."
        )
    )

    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help=("Whether to include sentence-transformers dense modules. "
              "It can be used for sentence-transformers models, like google/embeddinggemma-300m. "
              "Default these modules are not included.")
    )

    parser.add_argument(
        "--fuse-gate-up-exps", action="store_true",
        help="Fuse gate_exps and up_exps tensors into a single gate_up_exps tensor for MoE models.",
    )
    parser.add_argument(
        "--fp8-as-q8", action="store_true",
        help="Store tensors dequantized from FP8 as Q8_0 instead of BF16/F16.",
    )
    
    # Add change !
    parser.add_argument(
        "--monarch-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing fitted Monarch .pt or dependency-light .npz files. "
            "If set, the converter will keep the original dense tensors and additionally "
            "write Monarch tensors named *.monarch_l, *.monarch_r, *.monarch_perm."
        ),
    )

    parser.add_argument(
        "--monarch-dtype",
        type=str,
        choices=["f32", "f16"],
        default="f32",
        help="Data type used to store Monarch L/R tensors in GGUF. Use f32 for debugging.",
    )
    # change end !

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download
        allowed_patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"]
        if args.sentence_transformers_dense_modules:
            # include sentence-transformers dense modules safetensors files
            allowed_patterns.append("*.safetensors")
        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)

    if not dir_model.is_dir():
        logger.error(f'Error: {dir_model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    is_mistral_format = args.mistral_format
    if is_mistral_format and not _mistral_common_installed:
        raise ImportError(_mistral_import_error_msg)
    disable_mistral_community_chat_template = args.disable_mistral_community_chat_template

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        if not is_mistral_format:
            model_architecture = get_model_architecture(hparams, model_type)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                model_class = get_model_class(model_architecture, mmproj=(model_type == ModelType.MMPROJ))
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported")
                sys.exit(1)
        elif args.mmproj:
            assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
            from conversion.pixtral import PixtralModel
            model_class = PixtralModel
        elif "moe" in hparams:
            from conversion.mistral import MistralMoeModel
            model_class = MistralMoeModel
        else:
            from conversion.mistral import MistralModel
            model_class = MistralModel

        if args.mtp and args.no_mtp:
            logger.error("--mtp and --no-mtp are mutually exclusive")
            sys.exit(1)

        if args.mtp or args.no_mtp:
            from conversion.qwen import _Qwen35MtpMixin
            from conversion.step3 import Step35Model
            if not (issubclass(model_class, _Qwen35MtpMixin) or issubclass(model_class, Step35Model)):
                logger.error("--mtp / --no-mtp are only supported for Qwen3.5/3.6 and Step3.5 text variants today")
                sys.exit(1)
            if args.no_mtp:
                model_class.no_mtp = True
            if args.mtp:
                model_class.mtp_only = True

        model_instance = model_class(dir_model, output_type, fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split,
                                     remote_hf_model_id=hf_repo_id, disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                                     sentence_transformers_dense_modules=args.sentence_transformers_dense_modules,
                                     fuse_gate_up_exps=args.fuse_gate_up_exps,
                                     fp8_as_q8=args.fp8_as_q8,
                                     )
        # Add change !
        if args.monarch_dir is not None:
            if is_split:
                logger.warning(
                    "[Monarch] Split GGUF with extra Monarch tensors is not recommended "
                    "for the first debug version. Please use a single GGUF file first."
                )

            add_monarch_tensors_to_writer(
                model_instance=model_instance,
                monarch_dir=args.monarch_dir,
                monarch_dtype=args.monarch_dtype,
            )
                    
        # change end !



        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()
