import os
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk


# ============================================================
# 1. MonarchLinear: square version
# ============================================================

class MonarchLinear(nn.Module):
    """
    Square Monarch Linear prototype.

    This version supports only:
        in_features == out_features
        in_features % block_size == 0

    Structure:
        x -> blockdiag(R) -> permutation -> blockdiag(L)

    Stored parameters:
        R: [num_blocks, block_size, block_size]
        L: [num_blocks, block_size, block_size]
        perm: [hidden_size]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 64,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        init_std: float = 0.02,
    ):
        super().__init__()

        assert in_features == out_features, (
            f"Only square Linear is supported, got "
            f"in_features={in_features}, out_features={out_features}"
        )
        assert in_features % block_size == 0, (
            f"in_features={in_features} must be divisible by block_size={block_size}"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = in_features // block_size

        self.R = nn.Parameter(
            torch.empty(self.num_blocks, block_size, block_size, dtype=dtype)
        )
        self.L = nn.Parameter(
            torch.empty(self.num_blocks, block_size, block_size, dtype=dtype)
        )

        perm = self.make_monarch_permutation(in_features, block_size)
        self.register_buffer("perm", perm.long(), persistent=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

        self.init_std = init_std
        self.reset_parameters()

    @staticmethod
    def make_monarch_permutation(hidden_size: int, block_size: int) -> torch.Tensor:
        """
        Structured Monarch permutation.

        For hidden_size=4096, block_size=64:
            index shape [64, 64]
            transpose
            flatten
        """
        num_blocks = hidden_size // block_size
        idx = torch.arange(hidden_size)
        idx = idx.view(num_blocks, block_size)
        idx = idx.transpose(0, 1).contiguous().view(-1)
        return idx

    def reset_parameters(self):
        """
        Random init is usually better than pure identity for fitting arbitrary dense weights.

        Scale is intentionally small. During fitting, L/R learn the approximation.
        """
        with torch.no_grad():
            nn.init.normal_(self.R, mean=0.0, std=self.init_std)
            nn.init.normal_(self.L, mean=0.0, std=self.init_std)

    def block_diag_mul(self, x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """
        x:
            [tokens, hidden]

        blocks:
            [num_blocks, block_size, block_size]

        return:
            [tokens, hidden]
        """
        tokens = x.shape[0]
        x = x.view(tokens, self.num_blocks, self.block_size)

        y = torch.einsum("tbi,bij->tbj", x, blocks)

        return y.reshape(tokens, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        x = x.reshape(-1, self.in_features)

        # Use fp32 for fitting stability.
        x = x.float()
        R = self.R.float()
        L = self.L.float()

        x = self.block_diag_mul(x, R)
        x = x[:, self.perm]
        x = self.block_diag_mul(x, L)

        if self.bias is not None:
            x = x + self.bias.float()

        x = x.reshape(*original_shape[:-1], self.out_features)
        return x


# ============================================================
# 2. Model module helpers
# ============================================================

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    cur = model
    for attr in module_name.split("."):
        if attr.isdigit():
            cur = cur[int(attr)]
        else:
            cur = getattr(cur, attr)
    return cur


def find_square_linear_layers(
    model: nn.Module,
    block_size: int = 64,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """
    Find square nn.Linear layers whose dimensions are divisible by block_size.

    Default behavior:
        includes all square Linear layers.

    For LLaMA-2-7B, this typically includes:
        q_proj, k_proj, v_proj, o_proj
    """

    results = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        out_features, in_features = module.weight.shape

        if in_features != out_features:
            continue

        if in_features % block_size != 0:
            continue

        if include_keywords is not None:
            if not any(k in name for k in include_keywords):
                continue

        if exclude_keywords is not None:
            if any(k in name for k in exclude_keywords):
                continue

        results.append((name, module))

    return results


def print_all_linear_shapes(model: nn.Module):
    print("\n========== All Linear layers ==========")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name:70s} {tuple(module.weight.shape)}")
    print("=======================================\n")


# ============================================================
# 3. WikiText-2 loading
# ============================================================

def load_wikitext_texts(
    split: str = "train",
    max_samples: int = 128,
    min_chars: int = 20,
    dataset_disk_path: Optional[str] = None,
) -> List[str]:
    """
    Load WikiText-2 texts.

    Two modes:
    1. dataset_disk_path is None:
        load_dataset("wikitext", "wikitext-2-raw-v1")

    2. dataset_disk_path is not None:
        load_from_disk(dataset_disk_path)

    If your WSL has no internet, you can download/save the dataset elsewhere:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        ds.save_to_disk("./wikitext2_disk")
    Then run this script with:
        --dataset_disk_path ./wikitext2_disk
    """

    if dataset_disk_path is not None:
        print(f"[Dataset] Loading WikiText-2 from disk: {dataset_disk_path}")
        ds = load_from_disk(dataset_disk_path)
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            ds_split = ds[split]
        else:
            ds_split = ds
    else:
        print("[Dataset] Loading WikiText-2 from HuggingFace datasets...")
        ds_split = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    texts = []
    for item in ds_split:
        text = item["text"]
        if text is None:
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        if text.startswith("=") and text.endswith("="):
            continue

        texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"[Dataset] Loaded {len(texts)} calibration texts from WikiText-2.")
    return texts


# ============================================================
# 4. Tokenization into calibration batches
# ============================================================

def build_token_batches(
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 128,
    batch_size: int = 1,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build tokenized batches for model forward.

    For activation collection, smaller batch_size is usually safer.
    batch_size=1 is recommended for 7B if GPU memory is limited.
    """

    batches = []

    for start in range(0, len(texts), batch_size):
        sub_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            sub_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        batches.append(inputs)

    print(f"[Dataset] Built {len(batches)} token batches.")
    return batches


# ============================================================
# 5. Collect input activations for one target layer
# ============================================================

@torch.no_grad()
def collect_input_activations_for_layer(
    model: nn.Module,
    target_module_name: str,
    token_batches: List[Dict[str, torch.Tensor]],
    device: str,
    max_tokens_to_keep: int = 8192,
) -> torch.Tensor:
    """
    Collect input activations X for a given Linear layer.

    Return:
        X_cpu: [num_tokens, hidden], float32 on CPU
    """

    model.eval()

    target_module = get_module_by_name(model, target_module_name)
    activations = []

    def hook_fn(module, inputs, output):
        x = inputs[0].detach().float().cpu()
        x = x.reshape(-1, x.shape[-1])
        activations.append(x)

    handle = target_module.register_forward_hook(hook_fn)

    try:
        for inputs in token_batches:
            model(**inputs)

            current_tokens = sum(a.shape[0] for a in activations)
            if current_tokens >= max_tokens_to_keep:
                break
    finally:
        handle.remove()

    X = torch.cat(activations, dim=0)

    if X.shape[0] > max_tokens_to_keep:
        X = X[:max_tokens_to_keep]

    return X.contiguous()


# ============================================================
# 6. Fit one layer
# ============================================================

def fit_one_monarch_layer(
    dense_layer: nn.Linear,
    X_cpu: torch.Tensor,
    block_size: int,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    init_std: float,
    log_interval: int,
) -> Tuple[MonarchLinear, Dict[str, float]]:
    """
    Fit MonarchLinear to dense_layer on calibration activations X_cpu.
    """

    out_features, in_features = dense_layer.weight.shape

    monarch = MonarchLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        bias=dense_layer.bias is not None,
        dtype=torch.float32,
        init_std=init_std,
    ).to(device)

    dense_layer = dense_layer.to(device).eval()

    for p in dense_layer.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        monarch.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_samples = X_cpu.shape[0]

    best_rel_mse = float("inf")
    best_state = None

    pbar = tqdm(range(1, steps + 1), desc="fitting", leave=False)

    for step in pbar:
        idx = torch.randint(0, num_samples, (batch_size,))
        x = X_cpu[idx].to(device).float()

        with torch.no_grad():
            y_ref = dense_layer(x).float()

        y_hat = monarch(x).float()

        mse = F.mse_loss(y_hat, y_ref)
        ref_norm = y_ref.pow(2).mean().clamp_min(1e-8)
        rel_mse = mse / ref_norm

        optimizer.zero_grad(set_to_none=True)
        mse.backward()
        torch.nn.utils.clip_grad_norm_(monarch.parameters(), 1.0)
        optimizer.step()

        rel_mse_value = float(rel_mse.detach().cpu())

        if rel_mse_value < best_rel_mse:
            best_rel_mse = rel_mse_value
            best_state = {
                "L": monarch.L.detach().cpu().clone(),
                "R": monarch.R.detach().cpu().clone(),
                "perm": monarch.perm.detach().cpu().clone(),
                "bias": None if monarch.bias is None else monarch.bias.detach().cpu().clone(),
            }

        if step % log_interval == 0 or step == 1:
            pbar.set_postfix({
                "mse": f"{float(mse.detach().cpu()):.3e}",
                "rel": f"{rel_mse_value:.3e}",
            })

    if best_state is not None:
        with torch.no_grad():
            monarch.L.copy_(best_state["L"].to(device))
            monarch.R.copy_(best_state["R"].to(device))
            if monarch.bias is not None and best_state["bias"] is not None:
                monarch.bias.copy_(best_state["bias"].to(device))

    eval_metrics = evaluate_one_layer(
        dense_layer=dense_layer,
        monarch_layer=monarch,
        X_cpu=X_cpu,
        device=device,
        batch_size=batch_size,
    )

    eval_metrics["best_train_rel_mse"] = best_rel_mse

    return monarch, eval_metrics


# ============================================================
# 7. Evaluate one fitted layer
# ============================================================

@torch.no_grad()
def evaluate_one_layer(
    dense_layer: nn.Linear,
    monarch_layer: MonarchLinear,
    X_cpu: torch.Tensor,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    dense_layer = dense_layer.to(device).eval()
    monarch_layer = monarch_layer.to(device).eval()

    total_sq_error = 0.0
    total_sq_ref = 0.0
    total_abs_error = 0.0
    total_abs_ref = 0.0
    total_numel = 0

    for start in range(0, X_cpu.shape[0], batch_size):
        end = min(start + batch_size, X_cpu.shape[0])
        x = X_cpu[start:end].to(device).float()

        y_ref = dense_layer(x).float()
        y_hat = monarch_layer(x).float()

        diff = y_hat - y_ref

        total_sq_error += diff.pow(2).sum().item()
        total_sq_ref += y_ref.pow(2).sum().item()
        total_abs_error += diff.abs().sum().item()
        total_abs_ref += y_ref.abs().sum().item()
        total_numel += y_ref.numel()

    mse = total_sq_error / max(total_numel, 1)
    rel_mse = total_sq_error / max(total_sq_ref, 1e-12)
    mae = total_abs_error / max(total_numel, 1)
    rel_mae = total_abs_error / max(total_abs_ref, 1e-12)

    return {
        "mse": mse,
        "rel_mse": rel_mse,
        "mae": mae,
        "rel_mae": rel_mae,
    }


# ============================================================
# 8. Save fitted Monarch layer
# ============================================================

def safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "_")


def save_monarch_layer(
    monarch_layer: MonarchLinear,
    layer_name: str,
    save_dir: str,
    metrics: Dict[str, float],
):
    os.makedirs(save_dir, exist_ok=True)

    file_name = safe_layer_filename(layer_name) + ".pt"
    save_path = os.path.join(save_dir, file_name)

    obj = {
        "layer_name": layer_name,
        "in_features": monarch_layer.in_features,
        "out_features": monarch_layer.out_features,
        "block_size": monarch_layer.block_size,
        "num_blocks": monarch_layer.num_blocks,
        "L": monarch_layer.L.detach().cpu(),
        "R": monarch_layer.R.detach().cpu(),
        "perm": monarch_layer.perm.detach().cpu(),
        "bias": None if monarch_layer.bias is None else monarch_layer.bias.detach().cpu(),
        "metrics": metrics,
    }

    torch.save(obj, save_path)

    return save_path


# ============================================================
# 9. Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/c/model/llama-2-7b-hf",
        help="Local HuggingFace LLaMA model path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./monarch_params_llama2_7b_wikitext2",
        help="Directory to save fitted Monarch parameters.",
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--dataset_disk_path",
        type=str,
        default=None,
        help="Optional local dataset path saved by datasets.save_to_disk().",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=128,
        help="Number of WikiText-2 text samples.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Token length for each calibration text.",
    )
    parser.add_argument(
        "--token_batch_size",
        type=int,
        default=1,
        help="Batch size for model forward during activation collection.",
    )
    parser.add_argument(
        "--max_tokens_per_layer",
        type=int,
        default=8192,
        help="Maximum activation tokens used per layer.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--fit_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--init_std",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--include_keywords",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated keywords. Only layers containing these keywords will be fitted.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="For debugging. Fit only first N target layers.",
    )
    parser.add_argument(
        "--start_layer_idx",
        type=int,
        default=0,
        help="Start index in the discovered target layer list.",
    )

    parser.add_argument(
        "--print_shapes_only",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu. Default: auto.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Device] {device}")
    print(f"[Model] {args.model_path}")
    print(f"[Output] {args.output_dir}")

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------

    print("[Load] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print("[Load] Loading model...")
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    # Disable cache during activation collection.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad_(False)

    print_all_linear_shapes(model)

    if args.print_shapes_only:
        return

    # ------------------------------------------------------------
    # Find target square Linear layers
    # ------------------------------------------------------------

    include_keywords = None
    if args.include_keywords.strip():
        include_keywords = [x.strip() for x in args.include_keywords.split(",") if x.strip()]

    target_layers = find_square_linear_layers(
        model=model,
        block_size=args.block_size,
        include_keywords=include_keywords,
        exclude_keywords=None,
    )

    if args.start_layer_idx > 0:
        target_layers = target_layers[args.start_layer_idx:]

    if args.max_layers is not None:
        target_layers = target_layers[:args.max_layers]

    print("\n========== Target square Linear layers ==========")
    for i, (name, module) in enumerate(target_layers):
        print(f"[{i:03d}] {name:70s} {tuple(module.weight.shape)}")
    print(f"Total target layers: {len(target_layers)}")
    print("=================================================\n")

    # Save target layer list.
    target_list_path = os.path.join(args.output_dir, "target_layers.json")
    with open(target_list_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "index": i,
                    "name": name,
                    "shape": list(module.weight.shape),
                }
                for i, (name, module) in enumerate(target_layers)
            ],
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Load WikiText-2 and build token batches
    # ------------------------------------------------------------

    texts = load_wikitext_texts(
        split=args.dataset_split,
        max_samples=args.num_calib_samples,
        dataset_disk_path=args.dataset_disk_path,
    )

    token_batches = build_token_batches(
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.token_batch_size,
    )

    # ------------------------------------------------------------
    # Fit all target layers
    # ------------------------------------------------------------

    all_records = []

    for layer_idx, (layer_name, dense_layer) in enumerate(target_layers):
        print("\n" + "=" * 100)
        print(f"[Layer {layer_idx + 1}/{len(target_layers)}] {layer_name}")
        print(f"Weight shape: {tuple(dense_layer.weight.shape)}")
        print("=" * 100)

        # 1. Collect activations for this layer.
        print(f"[Activation] Collecting input activations for {layer_name}...")
        X_cpu = collect_input_activations_for_layer(
            model=model,
            target_module_name=layer_name,
            token_batches=token_batches,
            device=device,
            max_tokens_to_keep=args.max_tokens_per_layer,
        )

        print(f"[Activation] X shape = {tuple(X_cpu.shape)}")

        # 2. Fit Monarch parameters.
        monarch, metrics = fit_one_monarch_layer(
            dense_layer=dense_layer,
            X_cpu=X_cpu,
            block_size=args.block_size,
            device=device,
            steps=args.steps,
            batch_size=args.fit_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            init_std=args.init_std,
            log_interval=args.log_interval,
        )

        # 3. Save Monarch parameters.
        save_path = save_monarch_layer(
            monarch_layer=monarch,
            layer_name=layer_name,
            save_dir=args.output_dir,
            metrics=metrics,
        )

        # 4. Record metrics.
        record = {
            "layer_index": layer_idx,
            "layer_name": layer_name,
            "weight_shape": str(tuple(dense_layer.weight.shape)),
            "block_size": args.block_size,
            "num_blocks": dense_layer.in_features // args.block_size,
            "num_calib_tokens": int(X_cpu.shape[0]),
            "save_path": save_path,
            **metrics,
        }

        all_records.append(record)

        print("[Result]")
        print(f"  save_path          = {save_path}")
        print(f"  mse                = {metrics['mse']:.6e}")
        print(f"  rel_mse            = {metrics['rel_mse']:.6e}")
        print(f"  mae                = {metrics['mae']:.6e}")
        print(f"  rel_mae            = {metrics['rel_mae']:.6e}")
        print(f"  best_train_rel_mse = {metrics['best_train_rel_mse']:.6e}")

        # Save intermediate report after each layer.
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(args.output_dir, "fit_report.csv")
        json_path = os.path.join(args.output_dir, "fit_report.json")

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)

        # Release memory.
        del X_cpu
        del monarch
        torch.cuda.empty_cache() if device == "cuda" else None

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[Done] All target layers fitted.")
    print(f"[Report] {os.path.join(args.output_dir, 'fit_report.csv')}")
    print(f"[Report] {os.path.join(args.output_dir, 'fit_report.json')}")
    print("=" * 100)

    if len(all_records) > 0:
        df = pd.DataFrame(all_records)
        print("\n========== Summary ==========")
        print(df[[
            "layer_index",
            "layer_name",
            "rel_mse",
            "rel_mae",
            "save_path",
        ]])
        print("=============================\n")


if __name__ == "__main__":
    main()